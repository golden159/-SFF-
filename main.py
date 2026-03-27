from GUI import GUI
import serial_module

def main():
    try:
        gui = GUI()                 # 不再传入 camera
    except Exception as e:
        print(f"GUI初始化失败：{e}")
        return
    serial_module.gui = gui        # 供串口模块回调
    gui.root.mainloop()

if __name__ == "__main__":
    main()
