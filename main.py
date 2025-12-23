import sys
import os
import traceback
import logging
import numpy as np

# 配置日志输出，优化为INFO级别减少性能开销
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保numpy输出格式简洁易读
np.set_printoptions(precision=4, suppress=True)

# 中文显示由UI模块处理

# 添加当前目录到Python路径
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print(f"当前工作目录: {current_dir}")
    print(f"Python路径: {sys.path[:3]}")
except Exception as e:
    print(f"设置Python路径时出错: {e}")

# 添加全局调试函数（简化版）
def debug_print(title, message):
    """输出带标记的调试信息"""
    logger.debug(f"{title}: {message}")

# 导入PyQt5和UI组件
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt
    print("成功导入PyQt5模块")
except Exception as e:
    print(f"导入PyQt5模块失败: {e}")
    sys.exit(1)

# 导入项目模块
try:
    from ui.main_window import MainWindow
    print("成功导入UI模块")
except Exception as e:
    print(f"导入UI模块失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 移除了模型加载测试函数

def main():
    """主函数"""
    print("===== 程序启动 =====")
    
    # 在创建QApplication之前设置高DPI支持
    print("配置环境: 设置高DPI支持")
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        print("环境配置: AA_EnableHighDpiScaling 已启用")
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        print("环境配置: AA_UseHighDpiPixmaps 已启用")
    
    # 创建QApplication实例
    try:
        print("初始化Qt应用: 创建QApplication实例")
        app = QApplication(sys.argv)
        app.setApplicationName("牙列标记系统")
        app.setApplicationVersion("1.0.0")
        print("Qt应用初始化: QApplication实例创建成功")
        
        # 创建并显示主窗口
        try:
            print("创建主窗口: 初始化用户界面")
            window = MainWindow()
            print("主窗口创建: MainWindow实例创建成功")
            
            # 设置应用程序属性
            app.setQuitOnLastWindowClosed(False)
            print("应用程序配置: quitOnLastWindowClosed 设置为 False")
            
            # 显示主窗口
            print("界面显示: 显示主窗口")
            window.show()
            print(f"窗口状态: 主窗口显示成功，可见性: {window.isVisible()}")
            
            # 添加窗口关闭事件处理
            def on_window_closed():
                print("主窗口关闭事件被触发")
                app.quit()
            
            window.closeEvent = lambda event: on_window_closed()
            
            # 添加模型加载状态检查
            def check_model_loading():
                if hasattr(window, 'models'):
                    for model_type in ['maxilla', 'mandible']:
                        if model_type not in window.models or window.models[model_type] is None:
                            logger.info(f"未检测到{model_type}模型，建议用户导入")
                else:
                    logger.warning("模型存储属性未初始化")

            check_model_loading()
            
            # 运行应用程序主循环
            print("程序启动: 启动应用程序主循环")
            result = app.exec_()
            print(f"程序退出: 应用程序退出，退出码: {result}")
            sys.exit(result)
            
        except Exception as e:
            debug_print("主窗口错误", f"{str(e)}")
            traceback.print_exc()
            # 显示错误消息
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("主窗口创建失败")
            msg.setInformativeText(str(e))
            msg.setWindowTitle("错误")
            msg.exec_()
            sys.exit(1)
            
    except Exception as e:
        debug_print("应用启动失败", f"{str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()