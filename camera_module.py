from ctypes import c_ubyte

from _ctypes import addressof
from time import sleep

import gxipy as gx
from PIL import Image
import numpy as np
from gxipy.gxidef import *
from gxipy.ImageProc import Utility


class CameraHandler:
    def __init__(self):
        self.device_manager = gx.DeviceManager()
        self.cam = None
        self.image_convert = None
        self.is_opened = False  # 新增：标记相机状态
        self.camera_name = "未知相机"  # 相机名称

    def _get_best_valid_bits(self, pixel_format):
        """
        根据像素格式获取最佳有效位
        :param pixel_format: 像素格式
        :return: 有效位
        """
        if pixel_format in (GxPixelFormatEntry.MONO8,
                            GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8,
                            GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8,
                            GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8,
                            GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
            return DxValidBit.BIT0_7
        elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.MONO10_P,
                              GxPixelFormatEntry.BAYER_GR10, GxPixelFormatEntry.BAYER_RG10,
                              GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10,
                              GxPixelFormatEntry.BAYER_GR10_P, GxPixelFormatEntry.BAYER_RG10_P,
                              GxPixelFormatEntry.BAYER_GB10_P, GxPixelFormatEntry.BAYER_BG10_P,
                              GxPixelFormatEntry.BAYER_GR10_PACKED, GxPixelFormatEntry.BAYER_RG10_PACKED,
                              GxPixelFormatEntry.BAYER_GB10_PACKED, GxPixelFormatEntry.BAYER_BG10_PACKED):
            return DxValidBit.BIT2_9
        elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.MONO12_P,
                              GxPixelFormatEntry.BAYER_GR12, GxPixelFormatEntry.BAYER_RG12,
                              GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12,
                              GxPixelFormatEntry.BAYER_GR12_P, GxPixelFormatEntry.BAYER_RG12_P,
                              GxPixelFormatEntry.BAYER_GB12_P, GxPixelFormatEntry.BAYER_BG12_P,
                              GxPixelFormatEntry.BAYER_GR12_PACKED, GxPixelFormatEntry.BAYER_RG12_PACKED,
                              GxPixelFormatEntry.BAYER_GB12_PACKED, GxPixelFormatEntry.BAYER_BG12_PACKED):
            return DxValidBit.BIT4_11
        elif pixel_format in (GxPixelFormatEntry.MONO14, GxPixelFormatEntry.MONO14_P,
                              GxPixelFormatEntry.BAYER_GR14, GxPixelFormatEntry.BAYER_RG14,
                              GxPixelFormatEntry.BAYER_GB14, GxPixelFormatEntry.BAYER_BG14,
                              GxPixelFormatEntry.BAYER_GR14_P, GxPixelFormatEntry.BAYER_RG14_P,
                              GxPixelFormatEntry.BAYER_GB14_P, GxPixelFormatEntry.BAYER_BG14_P):
            return DxValidBit.BIT6_13
        elif pixel_format in (GxPixelFormatEntry.MONO16,
                              GxPixelFormatEntry.BAYER_GR16, GxPixelFormatEntry.BAYER_RG16,
                              GxPixelFormatEntry.BAYER_GB16, GxPixelFormatEntry.BAYER_BG16):
            return DxValidBit.BIT8_15
        return DxValidBit.BIT0_7

    def _convert_to_special_pixel_format(self, raw_image, pixel_format):
        """
        将原始图像转换为指定的像素格式
        :param raw_image: 原始图像
        :param pixel_format: 目标像素格式
        :return: 转换后的图像数组和缓冲区大小
        """
        if self.image_convert is None:
            print("Image convert object is not initialized.")
            return None, None
        self.image_convert.set_dest_format(pixel_format)
        valid_bits = self._get_best_valid_bits(raw_image.get_pixel_format())
        self.image_convert.set_valid_bits(valid_bits)

        buffer_out_size = self.image_convert.get_buffer_size_for_conversion(raw_image)
        output_image_array = (c_ubyte * buffer_out_size)()
        output_image = addressof(output_image_array)

        self.image_convert.convert(raw_image, output_image, buffer_out_size, False)
        if output_image is None:
            print('Pixel format conversion failed')
            return None, None
        return output_image_array, buffer_out_size

    def open_camera(self):
        try:
            dev_num, dev_info_list = self.device_manager.update_all_device_list()
            if dev_num == 0:
                print("未检测到相机设备")
                return False
            
            # 与camera.py一致的索引方式（从1开始）
            self.cam = self.device_manager.open_device_by_index(1)
            self.image_convert = self.device_manager.create_image_format_convert()
            
            # 获取相机名称
            try:
                dev_info = self.device_manager.get_device_info(1)
                self.camera_name = dev_info.get_device_model_name()
            except:
                self.camera_name = "工业相机"
            
            # 设置连续曝光模式
            remote_device = self.cam.get_remote_device_feature_control()
            
            # 设置曝光模式为连续模式
            try:
                exposure_mode_enum = remote_device.get_enum_feature("ExposureMode")
                exposure_mode_enum.set("Continuous")
                print("已设置曝光模式为连续模式")
            except Exception as e:
                print(f"设置曝光模式失败：{e}")
            
            # 关闭触发模式（避免干扰串口通信时序）
            try:
                trigger_mode_enum = remote_device.get_enum_feature("TriggerMode")
                trigger_mode_enum.set("Off")
                print("已关闭触发模式")
            except Exception as e:
                print(f"关闭触发模式失败：{e}")
            
            self.cam.stream_on()
            self.is_opened = True
            print("相机打开成功")
            return True
        except Exception as e:
            print(f"相机打开失败：{e}")
            self.close_camera()
            return False
        
    def capture_image(self):
        """
        捕获图像
        :return: 捕获的图像（如果成功），否则为 None
        """
        if self.cam is None:  # 直接判断 cam 是否为 None
            print("相机未打开")
            return None
        try:
            # 优化：减少采集间隔（从0.001秒减少到0.0001秒）
            sleep(0.0001)
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image is None:
                print("Getting image failed.")
                return None
            if raw_image.get_pixel_format() not in (
                    GxPixelFormatEntry.MONO8, GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
                mono_image_array, mono_image_buffer_length = self._convert_to_special_pixel_format(
                    raw_image, GxPixelFormatEntry.MONO8)
                if mono_image_array is None:
                    return None
                numpy_image = np.frombuffer(mono_image_array, dtype=np.ubyte, count=mono_image_buffer_length).reshape(
                    raw_image.frame_data.height, raw_image.frame_data.width)
            else:
                numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                return None
            img = Image.fromarray(numpy_image, 'L')
            # 优化：减少日志输出频率，只在调试时输出
            # print("Frame ID: %d   Height: %d   Width: %d" % (
            #     raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))
            return img
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None



    def close_camera(self):
        """关闭相机（强制释放资源）"""
        if self.cam is not None:
            try:
                self.cam.stream_off()
                self.cam.close_device()
            except Exception as e:
                print(f"关闭相机错误：{e}")

        self.cam = None
        self.image_convert = None
        self.is_opened = False
        print("相机资源已释放")