import os
import cv2
import numpy as np
import glob
import re
import time
import sys
from scipy.signal import find_peaks
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from functools import lru_cache

# 全局缓存对象，避免重复创建
_clahe_cache = None
_sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

def get_clahe():
    """获取缓存的CLAHE对象"""
    global _clahe_cache
    if _clahe_cache is None:
        _clahe_cache = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return _clahe_cache

# 尝试导入并行处理库（可选）
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# ---------- 鲁棒性最高的清晰度函数：Absolute Tenengrad ----------
def absolute_tenengrad(image, roi=None):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if roi:
        y_min, y_max, x_min, x_max = roi
        gray = gray[y_min:y_max, x_min:x_max]
        if gray.size == 0:
            return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.abs(gx) + np.abs(gy))


# ---------- 优化：适用于深孔下表面的聚焦度函数 ----------
def preprocess_image_for_deep_hole(image, roi=None):
    """
    深孔图像预处理函数（优化版本）
    针对深孔下表面图像进行增强处理
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if roi:
        y_min, y_max, x_min, x_max = roi
        gray = gray[y_min:y_max, x_min:x_max]
        if gray.size == 0:
            return None
    
    # 1. 直方图均衡化增强对比度（使用缓存对象）
    clahe = get_clahe()
    enhanced = clahe.apply(gray)
    
    # 2. 高斯滤波去噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 3. 锐化处理（使用预定义核）
    sharpened = cv2.filter2D(denoised, -1, _sharpen_kernel)
    
    # 4. 边缘增强（使用更高效的权重计算）
    edge_enhanced = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
  
    return edge_enhanced

def preprocess_bilateral_filter(image, roi=None):
    """
    双边滤波预处理算法
    保持边缘的同时去除噪声
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if roi:
        y_min, y_max, x_min, x_max = roi
        gray = gray[y_min:y_max, x_min:x_max]
        if gray.size == 0:
            return None
    
    # 1. CLAHE增强对比度
    clahe = get_clahe()
    enhanced = clahe.apply(gray)
    
    # 2. 双边滤波去噪（保持边缘）
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. 锐化处理
    sharpened = cv2.filter2D(denoised, -1, _sharpen_kernel)
    
    return sharpened

def anisotropy_focus_measure_with_preprocessing(image, roi=None, preprocess_func=preprocess_image_for_deep_hole):
    """
    各向异性聚焦度测量函数（支持不同预处理算法）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if roi:
        y_min, y_max, x_min, x_max = roi
        gray = gray[y_min:y_max, x_min:x_max]
        if gray.size == 0:
            return 0.0
    
    # 使用指定的预处理算法
    processed = preprocess_func(image, roi)
    if processed is None:
        return 0.0
    
    # 计算梯度
    gx = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值和方向
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    # 各向异性计算
    num_bins = 8
    bin_size = 2 * np.pi / num_bins
    
    directional_strength = np.zeros(num_bins)
    for i in range(num_bins):
        start_angle = i * bin_size - np.pi
        end_angle = (i + 1) * bin_size - np.pi
        
        mask = (direction >= start_angle) & (direction < end_angle)
        if np.any(mask):
            directional_strength[i] = np.mean(magnitude[mask])
    
    # 计算各向异性指标
    mean_strength = np.mean(directional_strength)
    if mean_strength > 0:
        anisotropy = np.std(directional_strength) / mean_strength
    else:
        anisotropy = 0
    
    return anisotropy

def anisotropy_focus_measure(image, roi=None):
    """
    各向异性聚焦度测量函数（优化版本）- 保持原有接口
    基于文献：各向异性聚焦度测量在非散瞳视网膜成像中表现优异
    参考：https://github.com/agmarrugo/anisotropy-focus
    """
    return anisotropy_focus_measure_with_preprocessing(image, roi, preprocess_image_for_deep_hole)


# ---------- 并行处理聚焦度计算函数 ----------
def calculate_focus_parallel(images, focus_func, roi=None, max_workers=4):
    """
    并行计算聚焦度值
    """
    if not PARALLEL_AVAILABLE or len(images) < 10:  # 图像数量少时不使用并行
        return [focus_func(img, roi) for img in images]
    
    def process_image(img):
        return focus_func(img, roi)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img) for img in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行聚焦度计算"):
            results.append(future.result())
    
    return results

def calculate_focus_parallel_with_preprocessing(images, roi=None, preprocess_func=preprocess_image_for_deep_hole, max_workers=4):
    """
    并行计算聚焦度值（支持不同预处理算法）
    """
    if not PARALLEL_AVAILABLE or len(images) < 10:
        return [anisotropy_focus_measure_with_preprocessing(img, roi, preprocess_func) for img in images]
    
    def process_image(img):
        return anisotropy_focus_measure_with_preprocessing(img, roi, preprocess_func)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img) for img in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行聚焦度计算"):
            results.append(future.result())
    
    return results


# ---------- 智能下表面识别算法（优化版本） ----------
def smart_bottom_focus_detection(indices, images, top_peak, roi, aspect_ratio=None):
    """
    智能下表面最佳对焦图识别算法（优化版本）
    使用各向异性聚焦度加权算法，专门针对深孔优化
    新增：根据深经比自动选择预处理策略
    """
    indices = np.array(indices)
    
    # 1. 智能分割上下表面
    diffs = np.diff(indices)
    if len(diffs) > 0:
        split_point = indices[np.argmax(diffs)]
    else:
        split_point = indices[len(indices)//2]
    
    # 2. 识别下表面图像组（使用布尔索引优化）
    bottom_mask = indices > split_point
    bottom_indices = indices[bottom_mask]
    bottom_images = [img for i, img in enumerate(images) if bottom_mask[i]]
    
    if len(bottom_indices) == 0:
        print("警告：未找到下表面图像组")
        return float(top_peak)
    
    # 3. 根据深经比自动选择预处理策略
    if aspect_ratio is not None:
        if aspect_ratio < 6.75:
            print(f"深经比 {aspect_ratio:.4f} < 6.75，使用双边滤波策略")
            preprocess_func = preprocess_bilateral_filter
        else:
            print(f"深经比 {aspect_ratio:.4f} >= 6.75，使用原始算法策略")
            preprocess_func = preprocess_image_for_deep_hole
    else:
        print("深经比未知，使用原始算法策略")
        preprocess_func = preprocess_image_for_deep_hole
    
    # 4. 计算下表面聚焦度（使用选定的预处理策略）
    focus_anisotropy = np.array(calculate_focus_parallel_with_preprocessing(
        bottom_images, roi, preprocess_func))
    
    # 5. 各向异性峰值检测和加权算法（优化版本）
    max_focus = np.max(focus_anisotropy)
    anisotropy_peaks, _ = find_peaks(focus_anisotropy, 
                                   height=max_focus * 0.15,
                                   distance=3,
                                   prominence=max_focus * 0.08)
    
    if len(anisotropy_peaks) > 0:
        # 各向异性有峰值时，使用多峰值加权平均
        anisotropy_peak_values = focus_anisotropy[anisotropy_peaks]
        anisotropy_peak_positions = bottom_indices[anisotropy_peaks]
        
        # 验证峰值质量：检查峰值是否在合理范围内
        max_peak_value = np.max(anisotropy_peak_values)
        valid_mask = anisotropy_peak_values >= max_peak_value * 0.7
        
        if np.any(valid_mask):
            valid_peak_values = anisotropy_peak_values[valid_mask]
            valid_peak_positions = anisotropy_peak_positions[valid_mask]
            
            # 计算峰值权重（基于聚焦度值）
            peak_weights = valid_peak_values / np.sum(valid_peak_values)
            
            # 加权平均计算最终位置
            anisotropy_best_position = np.sum(valid_peak_positions * peak_weights)
            
            # 直接使用加权平均位置，保留小数点后一位
            bottom_peak = round(anisotropy_best_position, 1)
        else:
            # 如果没有有效峰值，选择最大值
            anisotropy_max_idx = np.argmax(focus_anisotropy)
            bottom_peak = bottom_indices[anisotropy_max_idx]
        
    else:
        # 各向异性无峰值时，选择最大值
        anisotropy_max_idx = np.argmax(focus_anisotropy)
        bottom_peak = bottom_indices[anisotropy_max_idx]
    
    return float(bottom_peak)


# ---------- 自动 ROI（优化版本） ----------
def auto_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 10
    return max(y - pad, 0), min(y + h + pad, gray.shape[0]), \
        max(x - pad, 0), min(x + w + pad, gray.shape[1])


# ---------- 改进的 SNOW 直径计算（优化版本） ----------
def snow_diameter(img_bgr, pixel_size_um=0.1725):
    import porespy as ps
    from skimage import morphology, filters
    from skimage.measure import regionprops
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 使用缓存的CLAHE对象（为snow_diameter创建专用缓存）
    if not hasattr(snow_diameter, '_clahe_cache'):
        snow_diameter._clahe_cache = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    clahe = snow_diameter._clahe_cache
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary > 0
    binary = morphology.remove_small_objects(binary, min_size=20)
    binary = morphology.remove_small_holes(binary, area_threshold=20)

    snow = ps.filters.snow_partitioning(im=binary, r_max=5, sigma=3)
    regions = snow.regions * snow.im
    props = regionprops(regions)
    if not props:
        return None, []
    diameters = [prop.equivalent_diameter * pixel_size_um for prop in props]
    max_d = max(diameters) if diameters else None
    return max_d, diameters


# ---------- 圈图确认下表面直径功能 ----------
def click_and_circle_bottom_surface(img_bgr, px_size_um=0.1725, mag=1.0):
    """
    弹窗显示下表面最佳聚焦图，允许用户绘制多个圆并管理。
    功能：
    1. 点击两次确定圆的直径（第一次：边缘点，第二次：直径另一点）
    2. 按 Enter 确认当前圆（保存数据并清除当前框选线）
    3. 按 'd' 删除最近确认的圆
    4. 按 'c' 清除所有已确认圆
    5. 按 'q'、ESC 或关闭窗口退出并计算平均值
    6. 每条圆线有对应编号
    7. 记录每个确认圆的圆心坐标
    """
    original_img = img_bgr.copy()  # 保存原始图像
    clone = img_bgr.copy()
    confirmed_circles = []  # 存储已确认的圆 [(center, radius, num, diam_mm)]
    current_circle = None  # 当前正在绘制的圆 (center, radius)
    current_points = []  # 当前绘制的点
    circle_counter = 1  # 圆计数器
    drawing = False
    window_closed = False  # 跟踪窗口关闭状态

    def on_close():
        nonlocal window_closed
        window_closed = True

    def draw_circle(event, x, y, flags, param):
        nonlocal clone, confirmed_circles, current_circle, current_points, drawing, circle_counter

        if event == cv2.EVENT_LBUTTONDOWN:
            # 第一次点击：记录边缘点
            if len(current_points) == 0:
                current_points = [(x, y)]
                drawing = True
                # 绘制第一个点（绿色）
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("下表面直径测量工具", clone)
            # 第二次点击：记录直径上另一点
            elif len(current_points) == 1:
                current_points.append((x, y))
                drawing = True
                # 计算圆心和半径
                center = ((current_points[0][0] + current_points[1][0]) // 2,
                          (current_points[0][1] + current_points[1][1]) // 2)
                radius = int(np.sqrt((current_points[1][0] - current_points[0][0]) ** 2 +
                                     (current_points[1][1] - current_points[0][1]) ** 2)) // 2

                # 设置当前圆
                current_circle = (center, radius)

                # 更新显示
                draw_all_circles()

        elif event == cv2.EVENT_MOUSEMOVE and drawing and len(current_points) == 1:
            # 实时显示圆轮廓和十字线
            if len(current_points) == 1:
                center = ((current_points[0][0] + x) // 2, (current_points[0][1] + y) // 2)
                radius = int(np.sqrt((x - current_points[0][0]) ** 2 + (y - current_points[0][1]) ** 2)) // 2
                tmp = original_img.copy()  # 从原始图像开始

                # 标记第一个点（绿色）
                cv2.circle(tmp, current_points[0], 5, (0, 255, 0), -1)
                # 绘制当前圆和十字
                cv2.circle(tmp, center, radius, (255, 0, 0), 2)  # 蓝色表示未确认
                draw_center_cross(tmp, center, radius)
                cv2.imshow("下表面直径测量工具", tmp)

    def draw_center_cross(img, center, radius):
        """在圆心绘制延伸到边缘的蓝色十字线"""
        x, y = center
        # 绘制横线：从圆心向左延伸到圆边界，向右延伸到圆边界
        cv2.line(img, (x - radius, y), (x + radius, y), (255, 0, 0), 2)  # 蓝色
        # 绘制竖线：从圆心向上延伸到圆边界，向下延伸到圆边界
        cv2.line(img, (x, y - radius), (x, y + radius), (255, 0, 0), 2)  # 蓝色

    def draw_all_circles():
        """只绘制当前圆，不绘制已确认的圆"""
        nonlocal clone
        clone = original_img.copy()  # 从原始图像开始

        # 如果当前有正在绘制的圆，显示它（蓝色）
        if current_circle:
            center, radius = current_circle
            cv2.circle(clone, center, radius, (255, 0, 0), 2)  # 蓝色表示未确认
            draw_center_cross(clone, center, radius)
            # 标记两个原始点（绿色）
            cv2.circle(clone, current_points[0], 5, (0, 255, 0), -1)
            cv2.circle(clone, current_points[1], 5, (0, 255, 0), -1)

        cv2.imshow("下表面直径测量工具", clone)

    # 获取图像尺寸
    img_height, img_width = img_bgr.shape[:2]

    # 创建窗口并设置大小
    cv2.namedWindow("下表面直径测量工具", cv2.WINDOW_NORMAL)

    # 设置窗口大小
    window_width = 1024
    window_height = 768

    # 如果图像太大，则按比例缩小窗口
    if img_width > window_width or img_height > window_height:
        scale = min(window_width / img_width, window_height / img_height)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
    else:
        display_width = img_width
        display_height = img_height

    # 设置窗口大小
    cv2.resizeWindow("下表面直径测量工具", display_width, display_height)

    cv2.setMouseCallback("下表面直径测量工具", draw_circle)
    cv2.imshow("下表面直径测量工具", clone)

    # 设置窗口关闭处理
    cv2.setWindowProperty("下表面直径测量工具", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowTitle("下表面直径测量工具", "按ESC或关闭窗口退出")

    print("\n=== 下表面直径测量操作指南 ===")
    print("1. 点击两次确定圆的直径（第一次：边缘点，第二次：直径另一点）")
    print("2. 按 Enter 确认当前圆（保存数据并清除当前框选线）")
    print("3. 按 'd' 删除最近确认的圆")
    print("4. 按 'c' 清除所有已确认圆")
    print("5. 按 'q'、ESC 或关闭窗口退出并计算平均值")
    print("6. 每个确认圆的圆心坐标将被记录")

    while True:
        # 检查窗口是否已关闭
        if cv2.getWindowProperty("下表面直径测量工具", cv2.WND_PROP_VISIBLE) < 1:
            on_close()
            break

        key = cv2.waitKey(100) & 0xFF  # 添加短时间等待以避免高CPU使用
        if key == 255:  # 没有按键
            continue

        if key == 13:  # Enter键（确认当前圆）
            if current_circle:
                center, radius = current_circle
                diam_mm = 2 * radius * px_size_um * mag / 1000

                # 添加到已确认圆列表
                confirmed_circles.append((center, radius, circle_counter, diam_mm))
                # 打印圆心坐标
                print(
                    f"确认圆 #{circle_counter}: 圆心坐标 (x={center[0]}, y={center[1]}), 像素半径 = {radius} px, 物理直径 = {diam_mm:.4f} mm")

                # 重置当前状态
                current_circle = None
                current_points = []
                circle_counter += 1

                # 更新显示（清除所有轮廓）
                clone = original_img.copy()
                cv2.imshow("下表面直径测量工具", clone)
        elif key == ord('d') and confirmed_circles:  # 删除最近确认的圆
            removed = confirmed_circles.pop()
            print(f"删除圆 #{removed[2]}")
            # 更新显示
            clone = original_img.copy()
            cv2.imshow("下表面直径测量工具", clone)
        elif key == ord('c'):  # 清除所有圆
            confirmed_circles = []
            current_circle = None
            current_points = []
            circle_counter = 1
            clone = original_img.copy()
            cv2.imshow("下表面直径测量工具", clone)
            print("已清除所有圆")
        elif key == ord('q') or key == 27 or window_closed:  # q、ESC或窗口关闭
            break

    cv2.destroyAllWindows()

    # 计算并显示所有已确认圆的平均值
    if confirmed_circles:
        diameters = [diam for _, _, _, diam in confirmed_circles]
        avg_diam = np.mean(diameters)
        std_diam = np.std(diameters)

        print("\n===== 下表面直径测量结果 =====")
        # 输出每个圆的圆心坐标
        for (center, _, num, diam) in confirmed_circles:
            print(f"圆 #{num}: 圆心坐标 (x={center[0]}, y={center[1]}), 物理直径 = {diam:.4f} mm")
        print("------------------------")
        print(f"平均值: {avg_diam:.4f} mm")
        print(f"标准差: {std_diam:.4f} mm")
        print(f"测量次数: {len(confirmed_circles)}")
        print("========================")

        return avg_diam, confirmed_circles
    else:
        print("未确认任何圆")
        return None, []




class Tee:
    """同时向终端和文件写入"""

    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


# ---------- 主函数 ----------
def main():
    total_start_time = time.time()
    
    print("=== SFF微孔测量系统（性能优化版本）===")
    print(f"并行处理支持: {'是' if PARALLEL_AVAILABLE else '否'}")
    print("=" * 40)

    folder = filedialog.askdirectory(title="请选择图像文件夹")
    if not folder:
        print("未选择文件夹，程序退出")
        return

    save_path = os.path.join(folder, "results")
    os.makedirs(save_path, exist_ok=True)

    # 建立日志文件，以文件夹名字命名
    folder_name = os.path.basename(folder)
    log_path = os.path.join(save_path, f"{folder_name}_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)  # 重定向 print

    # 支持多种图像格式（优化版本）
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
    paths = []
    for ext in image_extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    
    # 按文件名中的数字排序（优化版本）
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    paths = sorted(paths, key=lambda x: extract_number(os.path.basename(x)))

    images, indices = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            idx = extract_number(os.path.basename(p))
            images.append(img)
            indices.append(idx)

    if not images:
        print("未找到有效图像")
        return

    roi = auto_roi(images[0]) or (0, images[0].shape[0], 0, images[0].shape[1])
    
    indices = np.array(indices)
    
    # 智能分割上下表面
    diffs = np.diff(indices)
    if len(diffs) > 0:
        split_point = indices[np.argmax(diffs)]
    else:
        split_point = indices[len(indices)//2]
    
    print(f"分割点: {split_point:.1f} μm")
    
    # 上表面识别（只分析分割点之前的上表面图组）
    print("开始计算上表面聚焦度...")
    focus_start = time.time()
    
    # 只计算上表面图组的聚焦度（优化版本）
    top_mask = indices <= split_point
    top_images = [img for i, img in enumerate(images) if top_mask[i]]
    top_indices = indices[top_mask]
    
    # 使用并行处理优化聚焦度计算
    focus_top = np.array(calculate_focus_parallel(top_images, absolute_tenengrad, roi))
    
    focus_end = time.time()
    print(f"上表面聚焦度计算完成，耗时：{focus_end - focus_start:.2f} 秒")
    
    # 上表面识别（使用SFF-main1.py的算法）
    peaks, _ = find_peaks(focus_top, height=np.max(focus_top) * 0.5, distance=10)
    if len(peaks) < 2:
        mid = top_indices[np.argmax(focus_top)]
        left_mask = top_indices <= mid
    else:
        left_mask = top_indices <= top_indices[sorted(peaks, key=lambda p: top_indices[p])[0]] + 50
    top_peak = float(top_indices[left_mask][np.argmax(focus_top[left_mask])])
    
    # 下表面识别（使用智能算法）
    print("\n开始智能下表面识别（深孔优化版本）...")
    
    # 先计算上表面直径用于深经比计算
    print("开始计算微孔直径...")
    top_d, _ = snow_diameter(images[np.argmin(np.abs(indices - top_peak))], pixel_size_um=0.1725)
    
    # 计算深经比（孔深/上表面直径）
    aspect_ratio = None
    if top_d is not None and top_d > 0:
        # 先估算孔深（使用简单的距离计算）
        estimated_depth = abs(indices[-1] - top_peak)  # 使用最大索引作为下表面估算
        aspect_ratio = estimated_depth / top_d
        print(f"估算深经比: {aspect_ratio:.4f}")
    
    # 使用智能算法进行下表面识别
    bottom_peak = smart_bottom_focus_detection(indices, images, top_peak, roi, aspect_ratio)    
    
    # 计算孔深（转换为毫米）
    depth_um = bottom_peak - top_peak
    depth_mm = depth_um / 1000.0
    
    # 重新计算实际深经比
    if top_d is not None and top_d > 0:
        actual_aspect_ratio = depth_um / top_d
        print(f"实际深经比: {actual_aspect_ratio:.4f}")
        if actual_aspect_ratio < 6.75:
            print("深经比 < 6.75，使用双边滤波策略")
        else:
            print("深经比 >= 6.75，使用原始算法策略")

    # ✅ 保存上表面最佳聚焦图
    top_idx = np.argmin(np.abs(indices - top_peak))
    cv2.imwrite(os.path.join(save_path, "top_focus_best.png"), images[top_idx])

    # ✅ 保存下表面最佳聚焦图
    bottom_idx = np.argmin(np.abs(indices - bottom_peak))
    cv2.imwrite(os.path.join(save_path, "bottom_focus_best.png"), images[bottom_idx])

    # 弹出下表面直径测量工具
    print("\n开始下表面直径手动测量...")
    print("即将弹出下表面最佳聚焦图，请手动圈选测量下表面直径")
    
    # 调用圈图工具测量下表面直径
    bottom_d_mm, bottom_circles = click_and_circle_bottom_surface(
        img_bgr=images[bottom_idx],
        px_size_um=0.1725,
        mag=1.0
    )
    
    # 转换上表面直径为毫米
    top_d_mm = None
    if top_d is not None:
        top_d_mm = top_d / 1000.0  # 转换为毫米
    
    # 计算平均直径
    avg_diameter_mm = None
    if top_d_mm is not None and bottom_d_mm is not None:
        avg_diameter_mm = (top_d_mm + bottom_d_mm) / 2.0
    elif top_d_mm is not None:
        avg_diameter_mm = top_d_mm
    elif bottom_d_mm is not None:
        avg_diameter_mm = bottom_d_mm
    
    # 计算深经比
    aspect_ratio = None
    if avg_diameter_mm is not None and avg_diameter_mm > 0:
        aspect_ratio = depth_mm / avg_diameter_mm

    # ===== 输出结果 =====
    print("===== 结果 =====")
    print(f"上表面聚焦图索引：{top_peak:.0f} μm")
    print(f"下表面聚焦图索引：{bottom_peak:.0f} μm")
    print(f"孔深：{depth_um:.1f} μm ({depth_mm:.4f} mm)")
    
    # 直径输出
    print(f"上表面直径：{top_d:.1f} μm" if top_d is not None else "上表面直径：无有效区域")
    print(f"下表面直径：{bottom_d_mm:.4f} mm" if bottom_d_mm is not None else "下表面直径：无有效测量")
    
    # 平均直径和深经比输出
    if avg_diameter_mm is not None:
        print(f"平均直径：{avg_diameter_mm:.4f} mm")
    if aspect_ratio is not None:
        print(f"深经比（孔深/上表面直径）：{aspect_ratio:.4f}")
        print(f"深经比阈值判断：{'< 6.75' if aspect_ratio < 6.75 else '>= 6.75'}")
        print(f"自动选择策略：{'双边滤波策略' if aspect_ratio < 6.75 else '原始算法策略'}")

    print(f"结果图已保存至：{save_path}")

    total_end = time.time()
    print(f"\n总运行时间：{total_end - total_start_time:.2f} 秒")

    log_file.close()  # 关闭日志文件
    sys.stdout = sys.__stdout__  # 恢复标准输出


if __name__ == "__main__":
    main()