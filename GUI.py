import os
import re
import socket
import time
import tkinter.font as tkFont
from time import sleep
from tkinter.ttk import *
from tkinter import *
from tkinter import filedialog
import serial
import serial.tools.list_ports
import threading
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import serial_module
from serial_module import uart_ctrl, uart_tx, ISHEX
from camera_module import CameraHandler
import tkinter as tk 
from serial_module import UART
from tkinter.ttk import Combobox

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("微芒干涉者 GUI")
        self.root.geometry("750x650+500+200")  # 恢复原始窗口大小
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 全局变量
        self.save_folder = None
        self.measurement_id = 1
        self.camera_on = False
        self.pzt_total_distance = 0.0  # 累计位移 mm
        self.cam_handler = CameraHandler()  # 相机句柄
        # 循环采集相关变量
        self.loop_idx = 1                               # 当前循环序号
        self.step_length = 0.0                          # 用于循环的步长
        self.use_delay = True  # True=按钮带延时，False=循环采集用
        self.second_origin_distance = None  # 新增：二次原点累计位移
        


        # ---------------- 服务器配置 ----------------
        self.server_ip = StringVar(value="127.0.0.1")
        self.server_port = StringVar(value="0")
        self.build_server_frame()

        # ---------------- 串口配置 ----------------
        self.build_serial_frame()

        # ---------------- 相机开关 ----------------
        self.build_camera_frame()

        # ---------------- 运动控制 ----------------
        self.build_control_frame()

        # ---------------- 图像采集 ----------------
        self.build_image_frame()

        # ---------------- 日志 ----------------
        self.build_log_frame()
        # 避免 AttributeError：给串口线程一个隐藏 Text 对象
        self.recv_text = Text()   # 不 pack/grid，仅作为占位
        self.server_tested = False  # 是否已测试连接
    # ========== 1. 服务器配置 ==========
    def build_server_frame(self):
        f = LabelFrame(self.root, text="服务器配置")
        f.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        Label(f, text="服务器IP:").grid(row=0, column=0, padx=5, pady=5)
        Entry(f, textvariable=self.server_ip, width=15).grid(row=0, column=1, padx=5)
        Label(f, text="端口:").grid(row=0, column=2, padx=5)
        Entry(f, textvariable=self.server_port, width=8).grid(row=0, column=3, padx=5)
        Button(f, text="测试连接", command=self.test_server_conn).grid(row=0, column=4, padx=5)

    # ========== 2. 串口配置 ==========
    def build_serial_frame(self):
        f = LabelFrame(self.root, text="PZT 串口配置")
        f.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        Label(f, text="串口号:").grid(row=0, column=0, padx=5, pady=5)
        Label(f, text="COM5").grid(row=0, column=1, padx=5, pady=5)
        Label(f, text="波特率:").grid(row=0, column=2, padx=5, pady=5)
        Label(f, text="9600").grid(row=0, column=3, padx=5, pady=5)
        self.uart_btn_text = StringVar(value="打开串口")
        Button(f, textvariable=self.uart_btn_text, width=12,
               command=self.toggle_uart).grid(row=0, column=4, padx=5, pady=5)

    # ========== 3. 相机开关 ==========
    def build_camera_frame(self):
        f = LabelFrame(self.root, text="相机控制")
        f.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        self.cam_btn_text = StringVar(value="打开相机")
        Button(f, textvariable=self.cam_btn_text, width=12,
               command=self.toggle_camera).grid(row=0, column=0, padx=5, pady=5)

    # ========== 4. 运动控制 ==========
    def build_control_frame(self):
        f = LabelFrame(self.root, text="运动控制")
        f.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')

        Button(f, text="REMOTE(+0.5mm)", width=18,
               command=lambda: self.move('left', 0.5,use_delay=True)).grid(row=0, column=0, padx=5, pady=5)
        Button(f, text="LOCAL(-0.5mm)", width=18,
               command=lambda: self.move('right', 0.5,use_delay=True)).grid(row=0, column=1, padx=5, pady=5)
        Button(f, text="REMOTE(+0.1mm)", width=18,
               command=lambda: self.move('left', 0.1,use_delay=True)).grid(row=1, column=0, padx=5, pady=5)
        Button(f, text="LOCAL(-0.1mm)", width=18,
               command=lambda: self.move('right', 0.1,use_delay=True)).grid(row=1, column=1, padx=5, pady=5)
        Button(f, text="REMOTE(+0.01mm)", width=18,
               command=lambda: self.move('left', 0.01,use_delay=True)).grid(row=2, column=0, padx=5, pady=5)
        Button(f, text="LOCAL(-0.01mm)", width=18,
               command=lambda: self.move('right', 0.01,use_delay=True)).grid(row=2, column=1, padx=5, pady=5)

        # 调整按钮布局
        Button(f, text="标记原点", width=18,
               command=lambda: (setattr(self, 'pzt_total_distance', 0.0),
                                self.log("✅ 标记原点完成，累计位移归零"))) \
            .grid(row=1, column=2, padx=5, pady=5)
        Button(f, text="返回原点", width=18,
               command=lambda: self.move(
                   'right' if self.pzt_total_distance > 0 else 'left',
                   abs(self.pzt_total_distance)
               )) \
            .grid(row=2, column=2, padx=5, pady=5)
        Button(f, text="标记二次原点", width=18, command=self.mark_second_origin).grid(row=1, column=3, padx=5, pady=5)
        Button(f, text="返回二次原点", width=18, command=self.return_second_origin).grid(row=2, column=3, padx=5, pady=5)
    # ========== 5. 图像采集 ==========
    def build_image_frame(self):
        f = LabelFrame(self.root, text="图像采集与上传")
        f.grid(row=4, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        Button(f, text="选择保存文件夹", command=self.select_folder).grid(row=0, column=0, padx=5, pady=5)
        Button(f, text="拍摄并上传", command=lambda: self.capture_and_upload_single(None, None)).grid(row=0, column=1, padx=5, pady=5)
        # —— 循环采集控件（照搬 GUI_backup） ——
        Label(f, text="循环次数:").grid(row=0, column=4, padx=5, pady=5)
        self.var_loop_count = StringVar(value="50")
        Entry(f, textvariable=self.var_loop_count, width=8).grid(row=0, column=5, padx=5, pady=5)
        Button(f, text="执行循环采集", width=12,command=self.execute_loop).grid(row=0, column=6, padx=5, pady=5)
        Label(f, text="步长(mm):").grid(row=0, column=2, padx=5, pady=5)
        self.var_step = StringVar(value="0.002")
        Entry(f, textvariable=self.var_step, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        # 快速采集模式选择
        self.fast_mode = BooleanVar(value=True)
        Checkbutton(f, text="快速采集模式", variable=self.fast_mode).grid(row=0, column=7, padx=5, pady=5)


    # ========== 6. 日志 ==========
    def build_log_frame(self):
        f = LabelFrame(self.root, text="操作日志")
        f.grid(row=5, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        self.log_text = Text(f, width=80, height=10)
        self.log_text.pack(padx=5, pady=5)
        Button(f, text="清空日志", command=lambda: self.log_text.delete(1.0, END)).pack(side=RIGHT)



    # ========== 通用方法 ==========
    def log(self, msg):
        self.log_text.insert(END, f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.log_text.see(END)

    # ----------- 串口 -----------
    def toggle_uart(self):
        if self.uart_btn_text.get() == "打开串口":
            ok = serial_module.uart_ctrl(1, "COM5", 9600)
            if ok:
                self.uart_btn_text.set("关闭串口")
                self.log("✅ PZT 串口已打开")
            else:
                messagebox.showerror("错误", "串口打开失败")
        else:
            serial_module.uart_ctrl(0, "", 0)
            self.uart_btn_text.set("打开串口")
            self.log("✅ PZT 串口已关闭")

    # ----------- 相机开关 -----------
    def toggle_camera(self):
        if not self.camera_on:
            if self.cam_handler.open_camera():
                self.camera_on = True
                self.cam_btn_text.set("关闭相机")
                self.log("✅ 相机已打开")
            else:
                messagebox.showerror("错误", "相机打开失败")
        else:
            self.cam_handler.close_camera()
            self.camera_on = False
            self.cam_btn_text.set("打开相机")
            self.log("✅ 相机已关闭")



    # ----------- 运动控制 -----------
    def move(self, direction, fixed_step=None, use_delay=False):
        if serial_module.UART is None or not serial_module.UART.is_open:
            messagebox.showerror("错误", "请先打开 PZT 串口")
            return

        step = fixed_step if fixed_step is not None else float(self.var_step.get())
        pulses = int(step * 16000)
        hex_pulses = f"{pulses:04X}"[2:4] + f"{pulses:04X}"[0:2]

        cmds = [
            f"000040015000{hex_pulses}000000",
            "0000400153000E000000",
            f"000040014400{'0000' if direction == 'left' else '0100'}0000",
            "00004001470000000000"
        ]
        
        # 优化：根据模式调整延时
        if use_delay:
            # 按钮模式：保持原有延时
            for cmd in cmds:
                serial_module.uart_tx(cmd, True)
                sleep(0.02)
            # 按钮模式加延时
            stall = {0.5: 1.5, 0.1: 0.7, 0.01: 0.3}.get(step, 0.3)
            sleep(stall)
        else:
            # 循环采集模式：减少延时
            for cmd in cmds:
                serial_module.uart_tx(cmd, True)
                sleep(0.005)  # 从0.02秒减少到0.005秒

        self.pzt_total_distance += step if direction == 'left' else -step
        self.log(f"{'右' if direction == 'right' else '左'}移 {step} mm，累计位移：{self.pzt_total_distance:.4f} mm")

    def on_closing(self):
        # 关闭相机
        if self.camera_on:
            self.cam_handler.close_camera()
            self.camera_on = False
        
        if abs(self.pzt_total_distance) > 1e-4:  # 允许微小误差
            messagebox.showwarning("提示", "没有回退原点，请回退原点后再退出")
        else:
            self.root.destroy()

    # ----------- 图片采集上传 -----------
    def select_folder(self):
        if abs(self.pzt_total_distance) > 1e-4:  # 允许微小误差
            messagebox.showwarning("提示", "请先标记原点")
            return

        self.save_folder = filedialog.askdirectory()
        if self.save_folder:
            self.measurement_id = self.get_next_measurement_id()
            self.log(f"保存文件夹：{self.save_folder}")

    def get_next_measurement_id(self):
        if not self.save_folder:
            return 1
        max_id = 0
        for d in os.listdir(self.save_folder):
            if d.startswith("measurement_"):
                try:
                    max_id = max(max_id, int(d.split("_")[1]))
                except:
                    pass
        return max_id + 1

    def remote_send(self, step_length):
        try:
            # 假设单步位移为 0.000125mm/脉冲
            pulse_per_step = 16000  # 1mm 对应的脉冲数
            total_pulses = int(step_length * pulse_per_step)
            hex_pulses = '{:04x}'.format(total_pulses)
            hex_pulses = hex_pulses[2:4] + hex_pulses[0:2]  # 转换为低位在前

            # 设置脉冲数的指令
            set_pulse_command = f'000040015000{hex_pulses}000000'
            # 发送设置脉冲数的指令
            uart_tx(set_pulse_command, True)
            # 等待一段时间，确保指令被设备接收和处理
            sleep(0.005)

            # 2. 设置移动速度（可选，如果需要设置速度的话）
            # 假设默认速度为 0x0E
            set_speed_command = '0000400153000E000000'
            uart_tx(set_speed_command, True)
            sleep(0.005)

            # 3. 发送执行移动的指令
            execute_command = '00004001470000000000'
            uart_tx(execute_command, True)


            # 发送脉冲指令（P命令），等待响应后再继续（无固定sleep）
            if not uart_tx(set_pulse_command, True) or not self.wait_for_response("50", timeout=1.0):
                raise Exception("脉冲指令未确认")
            
            # 发送速度指令（S命令），响应到达后立即执行下一步
            if not uart_tx(set_speed_command, True) or not self.wait_for_response("53", timeout=1.0):
                raise Exception("速度指令未确认")
            
            # 发送执行指令（G命令），无固定延时
            if not uart_tx(execute_command, True):
                raise Exception("执行指令发送失败")

        except Exception as e:
            self.log(f"❌ 左移失败：{e}")
            return False
    # 修改 local_send 方法（右移指令）
    def local_send(self, step_length):
        try:
            # 假设单步位移为 0.000125mm/脉冲
            pulse_per_step = 16000  # 1mm 对应的脉冲数
            total_pulses = int(step_length * pulse_per_step)
            hex_pulses = '{:04x}'.format(total_pulses)
            hex_pulses = hex_pulses[2:4] + hex_pulses[0:2]  # 转换为低位在前

            # 设置脉冲数的指令
            set_pulse_command = f'000040015000{hex_pulses}000000'
            # 发送设置脉冲数的指令
            uart_tx(set_pulse_command, True)
            # 等待一段时间，确保指令被设备接收和处理
            sleep(0.05)

            # 2. 设置移动速度（可选，如果需要设置速度的话）
            # 假设默认速度为 0x0E
            set_speed_command = '0000400153000E000000'
            uart_tx(set_speed_command, True)
            sleep(0.05)

            # 3. 发送执行移动的指令
            execute_command = '00004001470000000000'
            uart_tx(execute_command, True)


            # 发送脉冲指令，响应后立即继续
            if not uart_tx(set_pulse_command, True) or not self.wait_for_response("50", timeout=1.0):
                raise Exception("脉冲指令未确认")
            
            # 发送速度指令，无固定延时
            if not uart_tx(set_speed_command, True) or not self.wait_for_response("53", timeout=1.0):
                raise Exception("速度指令未确认")
            
            # 发送方向指令（D命令），响应后立即继续
            if not uart_tx(set_direction, True) or not self.wait_for_response("44", timeout=1.0):
                raise Exception("方向指令未确认")
            
            # 发送执行指令
            if not uart_tx(execute_command, True):
                raise Exception("执行指令发送失败")

        except Exception as e:
            self.log(f"❌ 右移失败：{e}")
    # 循环移动指令
    def loop_send(self):
        try:
            for i in range(2):
                # 左移
                # 设置脉冲数的指令
                set_pulse_command = '000040015000409C0000'
                # 发送设置脉冲数的指令
                if not uart_tx(set_pulse_command, True) or not self.wait_for_response("50", timeout=1.0):
                    raise Exception("左移脉冲指令未确认")

                # 设置方向指令（左移）
                set_direction_command = '00004001440000000000'
                if not uart_tx(set_direction_command, True) or not self.wait_for_response("44", timeout=1.0):
                    raise Exception("左移方向指令未确认")

                # 发送执行移动的指令
                execute_command = '00004001470000000000'
                if not uart_tx(execute_command, True):
                    raise Exception("左移执行指令发送失败")
                # 右移
                # 设置脉冲数的指令
                set_pulse_command = '000040015000409C0000'
                if not uart_tx(set_pulse_command, True) or not self.wait_for_response("50", timeout=1.0):
                    raise Exception("右移脉冲指令未确认")

                # 设置方向指令（右移）
                set_direction_command = '00004001440001000000'
                if not uart_tx(set_direction_command, True) or not self.wait_for_response("44", timeout=1.0):
                    raise Exception("右移方向指令未确认")

                # 发送执行移动的指令
                execute_command = '00004001470000000000'
                if not uart_tx(execute_command, True):
                    raise Exception("右移执行指令发送失败")

            self.mess_disp.insert(0.0, "✅ 循环移动指令执行完成\n")
        except Exception as e:
            self.mess_disp.insert(0.0, f"❌ 循环指令失败：{e}\n")
    # 选择本地保存文件夹
    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory()
        if self.save_folder:
            # 计算下一个测量编号
            self.measurement_id = self.get_next_measurement_id()
            self.mess_disp.insert(0.0, f"✅ 已选择保存文件夹：{self.save_folder}\n")
        else:
            self.mess_disp.insert(0.0, "⚠️ 未选择保存文件夹\n")

    def get_next_image_idx(self, measurement_folder):
        """根据累计位移计算图片序号，每0.001mm为1个单位"""
        idx = int(round(self.pzt_total_distance * 1000))
        return idx

    # 循环采集与上传
    # ============== 循环采集 ==============
    def execute_loop(self):
        # 前置条件：相机必须已打开
        if not self.camera_on:
            messagebox.showerror('提醒', '相机未打开，请打开相机后再试')
            return
        # 前置条件：累计位移不能为负
        if self.pzt_total_distance < 0:
            messagebox.showerror('提醒', '累计位移<0，请标记原点后再执行循环采集')
            return

        # 原有逻辑继续
        if not self.save_folder:
            messagebox.showerror('错误', '请先选择保存文件夹')
            return
        elif not serial_module.UART or not serial_module.UART.is_open:
            messagebox.showerror('错误', '请先打开PZT串口')
            return

        loop_cnt = int(self.var_loop_count.get())
        step_len = float(self.var_step.get())
        if step_len <= 0:
            messagebox.showerror('错误', '步长必须>0！')
            return

        measurement_folder = os.path.join(self.save_folder,
                                          f"measurement_{self.measurement_id}")
        os.makedirs(measurement_folder, exist_ok=True)
        
        # 性能统计
        start_time = time.time()
        self.log(f"🚀 开始循环采集：{loop_cnt}张图片，步长{step_len}mm")
        if self.fast_mode.get():
            self.log("⚡ 使用快速采集模式")
        else:
            self.log("📷 使用标准采集模式")

        # 优化：批量处理，减少延时
        for i in range(loop_cnt):
            # 左移（循环采集模式，减少延时）
            self.move('left', step_len, use_delay=False)
            
            # 优化：减少稳定时间
            sleep(0.02)  # 从0.1秒减少到0.02秒
            
            # 动态获取当前最大序号+1
            current_idx = self.get_next_image_idx(measurement_folder)
            
            # 根据模式选择采集方法
            if self.fast_mode.get():
                success = self.capture_and_upload_single_fast(measurement_folder, current_idx)
            else:
                success = self.capture_and_upload_single(measurement_folder, current_idx)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                self.log(f"📊 进度：{i + 1}/{loop_cnt} ({((i + 1) / loop_cnt * 100):.1f}%)")
            
            if not success:
                self.log(f"❌ 第{i + 1}张图片采集失败，继续下一张")
        
        # 性能统计结果
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_image = total_time / loop_cnt
        images_per_second = loop_cnt / total_time
        
        self.log(f"✅ 循环采集完成！")
        self.log(f"📊 总耗时：{total_time:.2f}秒")
        self.log(f"📊 平均每张：{avg_time_per_image:.3f}秒")
        self.log(f"📊 采集速度：{images_per_second:.1f}张/秒")

    def capture_and_upload_single_fast(self, measurement_folder=None, current_idx=None):
        """快速采集模式 - 优化版本"""
        if measurement_folder is None:
            measurement_folder = os.path.join(self.save_folder, f"measurement_{self.measurement_id}")
            os.makedirs(measurement_folder, exist_ok=True)
        if current_idx is None:
            current_idx = self.get_next_image_idx(measurement_folder)
        
        try:
            # 优化1：减少缓冲区清理帧数（从10帧减少到3帧）
            if hasattr(self.cam_handler, 'flush_buffer'):
                self.cam_handler.flush_buffer()
            else:
                # 快速清理缓冲区
                for _ in range(3):  # 从10帧减少到3帧
                    self.cam_handler.capture_image()

            # 优化2：减少稳定时间（从0.05秒减少到0.01秒）
            time.sleep(0.01)

            # 拍摄图像
            img = self.cam_handler.capture_image()
            if not img:
                return False

            # 优化3：使用更快的保存格式和压缩
            local_path = os.path.join(measurement_folder, f"image_{current_idx}.jpg")
            img.save(local_path, 'JPEG', quality=95, optimize=True)  # 使用JPEG格式，更快

            # 优化4：异步上传，不阻塞主流程
            if self.server_tested:
                threading.Thread(
                    target=self.upload_to_server,
                    args=(local_path, "measurement", f"image_{current_idx}.jpg"),
                    daemon=True
                ).start()
            
            return True

        except Exception as e:
            return False

    def capture_and_upload_single(self, measurement_folder=None, current_idx=None):
        """标准采集模式 - 原版本"""
        if measurement_folder is None:
            measurement_folder = os.path.join(self.save_folder, f"measurement_{self.measurement_id}")
            os.makedirs(measurement_folder, exist_ok=True)
        if current_idx is None:
            current_idx = self.get_next_image_idx(measurement_folder)  # 内部获取序号
        try:
            # 1. 清空相机缓冲区，丢弃PZT移动过程中产生的中间帧
            # （假设相机对象支持flush_buffer方法，若不支持可通过连续读取实现）
            if hasattr(self.cam_handler, 'flush_buffer'):
                self.cam_handler.flush_buffer()  # 清空缓冲区
            else:
                # 若相机无flush_buffer，通过连续读取丢弃缓存帧
                for _ in range(10):  # 读取10帧丢弃，确保拿到最新帧
                    self.cam_handler.capture_image()

            # 移动后稳定时间
            time.sleep(0.05)

            # 拍摄图像
            img = self.cam_handler.capture_image()  # 假设返回图像对象
            if not img:
                self.log("❌ 拍摄失败：未获取到图像")
                return False

            # 保存图像（示例：假设img为PIL.Image对象）
            local_path = os.path.join(measurement_folder, f"image_{current_idx}.png")
            img.save(local_path)
            self.log(f"✅ 保存成功：{local_path}")

            # 上传逻辑（示例：假设upload_to_server为异步方法）
            threading.Thread(
                target=self.upload_to_server,
                args=(local_path, "measurement", f"image_{current_idx}.png"),
                daemon=True
            ).start()
            return True

        except Exception as e:  # 新增：捕获所有异常并处理
            self.log(f"❌ 采集过程出错：{str(e)}")
            return False

    # 上传到云服务器
    def upload_to_server(self, local_path, folder_name, filename):
        """如果未测试连接，静默返回，不上传也不提示"""
        if not self.server_tested:
            return  # 静默跳过上传

        try:
            ip = self.server_ip.get().strip()
            port = int(self.server_port.get().strip())

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((ip, port))

            sock.send(folder_name.encode())
            if sock.recv(1024) != b'ACK1':
                return

            sock.send(filename.encode())
            if sock.recv(1024) != b'ACK2':
                return

            with open(local_path, 'rb') as f:
                while True:
                    data = f.read(1024)
                    if not data:
                        break
                    sock.send(data)

            if sock.recv(1024) == b'OK':
                self.log(f"✅ 上传成功：{filename}")
            else:
                self.log("❌ 上传失败：服务器未返回 OK")

            sock.close()

        except Exception:
            # 静默忽略所有上传异常
            pass

    # ========== 服务器测试连接 ==========
    def test_server_conn(self):
        try:
            ip = self.server_ip.get()
            port = int(self.server_port.get())
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((ip, port))
            sock.close()
            self.log("✅ 服务器连接成功")
            self.server_tested = True  # 标记为已测试
        except Exception as e:
            self.log(f"❌ 服务器连接失败：{e}")
            self.server_tested = False

    # 清空日志
    def txt_mess_clr(self):
        self.mess_disp.delete(1.0, END)

    def mark_second_origin(self):
        self.second_origin_distance = self.pzt_total_distance
        self.log(f"✅ 已标记二次原点，累计位移：{self.second_origin_distance:.4f} mm")

    def return_second_origin(self):
        if self.second_origin_distance is None:
            self.log("⚠️ 尚未标记二次原点")
            return
        delta = self.second_origin_distance - self.pzt_total_distance
        if abs(delta) < 1e-4:
            self.log("✅ 已在二次原点，无需移动")
            return
        direction = 'left' if delta > 0 else 'right'
        self.move(direction, abs(delta), use_delay=True)
        self.pzt_total_distance = self.second_origin_distance
        self.log(f"✅ 已返回二次原点，累计位移：{self.pzt_total_distance:.4f} mm")


if __name__ == '__main__':
    gui = GUI()
    gui.root.mainloop()