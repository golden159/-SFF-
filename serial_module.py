# flake8: noqa  # 关闭整文件 PEP8 警告
from time import sleep          # 消除 “未使用 sleep”
import serial
import serial.tools.list_ports
import threading
from tkinter import messagebox

# 全局变量先声明类型，消除 “未定义” 警告
UART: serial.Serial | None = None
RX_THREAD: threading.Thread | None = None
gui_recv_callback = None            # type: ignore


def ISHEX(data: str) -> bool:
    if len(data) % 2:
        return False
    return all(ch in '0123456789ABCDEFabcdef' for ch in data)


def _open_port(port: str, baud: int) -> serial.Serial:
    """真正打开端口的私有函数，消除 is_open / close 的 None 警告"""
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.01,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
    )
    return ser


def uart_ctrl(fun: int, port: str, baud: int) -> bool:
    global UART, RX_THREAD
    if fun == 1:  # 打开
        try:
            if UART is not None and UART.is_open:
                UART.close()
            UART = _open_port(port, baud)
            lock = threading.Lock()
            RX_THREAD = UartRxThread('URX1', lock)
            RX_THREAD.daemon = True
            RX_THREAD.start()
            RX_THREAD.resume()
            return True
        except Exception as e:
            print("串口打开错误：", e)
            return False
    else:  # 关闭
        if RX_THREAD is not None:
            RX_THREAD.pause()
        if UART is not None and UART.is_open:
            UART.close()
        UART = None
        print("串口已关闭")
        return True


def uart_tx(data: str, is_hex: bool = False) -> bool:
    if UART is None or not UART.is_open:
        messagebox.showerror('错误', '串口未打开')
        return False

    try:
        if is_hex:
            data = data.ljust(20, '0')[:20]
            if not ISHEX(data):
                messagebox.showerror('错误', '无效的十六进制指令')
                return False
            UART.write(bytes.fromhex(data))
        else:
            UART.write(data.encode('gb2312'))
        return True
    except Exception as e:
        messagebox.showerror('错误', f'发送失败：{e}')
        return False


class UartRxThread(threading.Thread):  # 类名大写，符合 PEP8
    def __init__(self, name: str, lock: threading.Lock) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.lock = lock
        self._event = threading.Event()
        self._event.set()

    def run(self) -> None:
        print('开启数据接收线程')
        while True:
            self._event.wait()
            if UART is not None and UART.is_open:
                try:
                    buf = UART.read_all()
                    if buf:
                        hex_data = buf.hex().upper()
                        print("收到数据：", hex_data)
                        if gui_recv_callback is not None:
                            gui_recv_callback(hex_data)  # type: ignore
                except Exception as e:
                    print("接收错误：", e)
            sleep(0.001)

    def pause(self) -> None:
        self._event.clear()

    def resume(self) -> None:
        self._event.set()
