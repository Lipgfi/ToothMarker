import sys

import os

import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox

from PyQt5.QtCore import Qt, pyqtSignal

import pyvista as pv

from pyvistaqt import QtInteractor

import open3d as o3d



# 配置基本日志输出，确保所有调试信息可见

np.set_printoptions(precision=4, suppress=True)



# 添加项目根目录到Python路径

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





class ModelViewer(QWidget):
    """3D模型查看器组件"""
    

    # 信号定义

    model_loaded = pyqtSignal(str)  # 模型加载完成信号

    model_error = pyqtSignal(str)   # 模型加载错误信号

    fit_plane_completed = pyqtSignal(object)  # 平面拟合完成信号，传递平面参数

    marked_points_updated = pyqtSignal(list)  # 标记点更新信号，传递标记点列表

    

    def __init__(self, parent=None):

        super().__init__(parent)

        # 添加新的配置参数

        self.background_color = 'lightgray'  # 默认背景颜色

        self.show_grid = True  # 是否显示网格

        self.show_axes = True  # 是否显示坐标轴

        self._marking_mode = "plane"  # "plane" "maxilla" "incisive_papilla"

        self.init_ui()

        self.reset_data()

    

    def init_ui(self):
        """初始化用户界面"""

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建PyVista交互器
        try:
            self.plotter = QtInteractor(self)
            self.layout.addWidget(self.plotter)
            
            # 设置背景色
            self.plotter.set_background(self.background_color)  

            

            # 添加坐标轴和网格

            print("[MODEL_VIEWER] 添加坐标轴和网格...")

            self._setup_visualization()

            

            # 添加交互帮助文本

            self._add_help_text()

            

            # 设置为XY视图

            print("[MODEL_VIEWER] 设置为XY视图...")

            self.plotter.view_xy()

            

            # 标记功能将在用户点击"开始标记按钮时启用

            print("[MODEL_VIEWER] 标记功能已准备就绪，等待用户启用...")

            

            # 执行初始渲染

            print("[MODEL_VIEWER] 执行初始渲染...")

            self.plotter.render()

            print("[MODEL_VIEWER] 3D查看器初始化完成")

            

        except Exception as e:

            error_label = QLabel(f"初始化3D查看器失败 {str(e)}")

            error_label.setAlignment(Qt.AlignCenter)

            error_label.setStyleSheet("color: red; font-weight: bold;")

            self.layout.addWidget(error_label)

            print(f"[MODEL_VIEWER ERROR] 初始化3D查看器失败 {e}")

            import traceback

            traceback.print_exc()

    

    def reset_data(self):
        """重置所有数据状态"""

        # 重置模型相关数据

        self.models = {}

        self.original_models = {}  # 保存原始模型（未简化的模型

        self.model_actors = {}

        self.contact_points_actor = None

        

        # 保留平面相关数据，仅重置平面演员，确保标记点和平面状态不丢失

        if not hasattr(self, 'marked_points'):

            self.marked_points = []

        if not hasattr(self, 'plane_fitted'):

            self.plane_fitted = False

        if not hasattr(self, 'plane_params'):

            self.plane_params = None

        self.plane_actor = None  # 仅重置平面演员，保留平面参数

        

        # 重置标记点演员，但保留标记点数据

        self.marked_points_actor = None

        if hasattr(self, 'marked_lines_actor'):

            self.marked_lines_actor = None

        

        # 初始化marker_lines属性，用于存储标记线供投影使用

        self.marker_lines = []

        

        self.current_transparency = 0.0  # 默认透明度，降低透明度以提高清晰度

        

        # 重置模型路径和名称

        self.current_model_path = None

        self.current_model_name = None

        

        # 重置模型质心

        self.model_centroid = None

        

        # 重置接触点数量

        self.contact_points = None

        

        print("[MODEL_VIEWER] 数据已重置，标记点和平面参数已保存")

        self.contact_normals = None

        

        # 重置标记点数量

        self.marked_points = []

        self.marked_points_modes = []  # 记录每个标记点的来源模式

        self.marked_lines_actor = None  # 用于存储标记点之间的连线

        self.marker_lines = []  # 存储标记线的3D坐标 [(p1, p2), (p3, p4), ...]

        

        # 重置平面数据

        self.plane_params = None

        self.plane_fitted = False

        



        

        # 重置切牙乳突相关数据

        self.incisive_papilla_point = None  # 切牙乳突前

        self.incisive_papilla_anterior_point = None  # 切牙乳突前-10mm位置�?

        self.symmetry_line_actor = None  # 对称线演员

        self.cutting_plane_actor = None  # 切割平面演员

        

        # 重置maxilla垂直平面相关数据

        if hasattr(self, 'maxilla_vertical_plane_actor'):

            self.maxilla_vertical_plane_actor = None

        if hasattr(self, 'maxilla_vertical_plane_params'):

            self.maxilla_vertical_plane_params = None

        

        # 重置下颌后槽牙槽嵴相关数量

        if hasattr(self, 'mandible_crest_points'):

            self.mandible_crest_points = []

        if hasattr(self, 'mandible_crest_line_actor'):

            self.mandible_crest_line_actor = None

        if hasattr(self, 'mandible_crest_projection_actor'):

            self.mandible_crest_projection_actor = None

        

        # 重置牙槽嵴相关数量
        if hasattr(self, 'alveolar_ridge_points'):
            self.alveolar_ridge_points = []
        if hasattr(self, 'alveolar_ridge_curve_actor'):
            self.alveolar_ridge_curve_actor = None
        
        # 重置划分上颌相关数据
        if hasattr(self, 'divide_maxilla_points'):
            self.divide_maxilla_points = []
        if hasattr(self, 'divide_maxilla_curve'):
            self.divide_maxilla_curve = []
        if hasattr(self, 'divide_maxilla_line_actor'):
            self.divide_maxilla_line_actor = None

        

        # 不再直接操作父窗口的特征点数量

        # 特征点的管理完全由MainWindow类负�?

        print("[CRITICAL DEBUG] 数据重置完成")

        

    def _setup_visualization(self):
        """设置可视化参数"""

        # 添加坐标轴

        if self.show_axes:

            self.plotter.add_axes()

            

        # 添加网格

        if self.show_grid:

            # 使用更清晰的网格设置，移除可能不支持的opacity参数

            self.plotter.show_grid(

                grid='front',

                location='outer',

                color='gray',

                xlabel='X (mm)',

                ylabel='Y (mm)',

                zlabel='Z (mm)',

                minor_ticks=True

            )

            

        # 暂时移除光源设置，避免初始化错误

        # 后续可以使用PyVista支持的方式重新实现光源功�?

    

    def _add_help_text(self):

        """添加帮助文本"""

        help_text = """
        操作说明：
        - 左键拖动：旋转视图
        - 中键拖动：平移视图
        - 滚轮：缩放视图
        - 右键点击模型：标记点（最多10个）
        """

        self.plotter.add_text(help_text, position='upper_left', font_size=10)

        print("[MODEL_VIEWER] 添加操作帮助文本")

    

    def set_orthographic_projection(self):

        """设置正交投影视图"""

        # 设置相机为正交投影模�?

        camera = self.plotter.camera

        camera.parallel_projection = True

        # 设置合适的平行缩放以确保整个模型可见

        self.plotter.reset_camera()

        print("[MODEL_VIEWER] 已切换到正交投影视图")

        return camera

        

    def set_perspective_projection(self):

        """设置透视投影视图"""

        # 设置相机为透视投影模式

        camera = self.plotter.camera

        camera.parallel_projection = False

        print("[MODEL_VIEWER] 已切换到透视投影视图")

        return camera

        

    def get_orthographic_view_image(self, width=800, height=600):

        """获取正交投影视图的图�?

        

        Args:

            width: 图像宽度

            height: 图像高度

            

        Returns:

            numpy数组: 正交投影视图的图像数量

        """

        # 保存当前相机状态

        original_parallel = self.plotter.camera.parallel_projection

        original_position = self.plotter.camera.position

        original_focal = self.plotter.camera.focal_point

        original_viewup = self.plotter.camera.view_up

        

        # 设置正交投影

        self.set_orthographic_projection()

        

        # 渲染并获取图�?

        self.plotter.render()

        image = self.plotter.screenshot(None, window_size=(width, height), return_img=True)

        

        # 恢复原始相机状态

        if not original_parallel:

            self.set_perspective_projection()

        self.plotter.camera.position = original_position

        self.plotter.camera.focal_point = original_focal

        self.plotter.camera.view_up = original_viewup

        

        return image

    

    def add_model(self, model_name, mesh, color=None, model_type=None, transform=None):

        """添加3D模型到查看器

        

        Args:

            model_name: 模型名称

            mesh: Open3D网格对象或PyVista网格对象

            color: 模型颜色

            model_type: 模型类型（maxilla, mandible, occlusion�?

            transform: 变换矩阵 (4x4 numpy数组)，用于同步应用坐标变�?

        """

        print("\n[CRITICAL DEBUG] ====== 添加模型开�?======")

        print(f"[CRITICAL DEBUG] 模型名称: {model_name}")

        print(f"[CRITICAL DEBUG] 模型类型: {type(mesh)}")

        print(f"[CRITICAL DEBUG] 传入的模型类型标�? {model_type}")

        print(f"[CRITICAL DEBUG] 变换矩阵: {transform is not None}")

        

        # 添加plotter存在性检查
        if not hasattr(self, 'plotter') or self.plotter is None:
            print(f"[CRITICAL ERROR] 3D查看器未初始化！无法添加模型: {model_name}")
            return False
        
        # 检查mesh是否为None
        if mesh is None:
            print(f"[CRITICAL ERROR] 模型为None！无法添加")
            return False
            
        # 确保models和model_actors字典存在
        if not hasattr(self, 'models'):
            self.models = {}
        if not hasattr(self, 'model_actors'):
            self.model_actors = {}

        

        try:

            # 保存原始模型

            self.models[model_name] = mesh

            print(f"[CRITICAL DEBUG] 原始模型已保存到字典")

            

            # 使用优化的颜色方向

            if color is None:

                # 使用专业且易区分的颜色方向

                colors = {

                    'maxilla': [1, 0.5, 0.5],      # 粉红�?- 上颌

                    'mandible': [0.5, 1, 0.5],     # 浅绿�?- 下颌

                    'occlusion': [0.5, 0.5, 1],     # 浅蓝�?- 咬合

                    'debug_cube': [1, 1, 0.8]       # 浅黄�?- 调试立方向

                }

                color = colors.get(model_type, [1, 1, 0.8])  # 默认使用浅黄�?

            print(f"[CRITICAL DEBUG] 最终使用的模型颜色: {color}")

            

            # 简化网格转换逻辑

            pv_mesh = None

            if isinstance(mesh, o3d.geometry.TriangleMesh):

                print(f"[CRITICAL DEBUG] 处理Open3D TriangleMesh...")

                

                # 检查网格基本属�?

                try:

                    num_vertices = len(mesh.vertices)

                    num_triangles = len(mesh.triangles)

                    print(f"[CRITICAL DEBUG] Open3D网格 - 顶点�? {num_vertices}, 三角面数: {num_triangles}")

                    

                    if num_vertices == 0 or num_triangles == 0:

                        print(f"[CRITICAL ERROR] 网格数据无效 - 顶点数或三角面数量")

                        return False

                    

                    # 提取数据

                    vertices = np.asarray(mesh.vertices)

                    triangles = np.asarray(mesh.triangles)

                    print(f"[CRITICAL DEBUG] 顶点数组形状: {vertices.shape}")

                    print(f"[CRITICAL DEBUG] 三角面数组形�? {triangles.shape}")

                    

                    # 应用坐标转换（如果提供）

                    if transform is not None:

                        print(f"[CRITICAL DEBUG] 应用变换矩阵: {transform}")

                        try:

                            # 确保变换矩阵�?x4格式

                            if transform.shape != (4, 4):

                                print(f"[CRITICAL WARNING] 变换矩阵形状错误，期�?4,4)，得到{transform.shape}")

                                # 尝试�?x3旋转矩阵扩展�?x4

                                if transform.shape == (3, 3):

                                    ext_transform = np.eye(4)

                                    ext_transform[:3, :3] = transform

                                    transform = ext_transform

                                    print(f"[CRITICAL DEBUG] 已将3x3旋转矩阵扩展�?x4")

                                else:

                                    print(f"[CRITICAL WARNING] 无法处理的变换矩阵形式")

                             

                            # 转换为齐次坐标并应用变换

                            vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])

                            vertices_transformed = (transform @ vertices_homogeneous.T).T[:, :3]

                            vertices = vertices_transformed

                            print(f"[CRITICAL DEBUG] 坐标变换应用成功")

                            

                            # 如果是上颌或下颌模型且存在平面，需要同步更新平面



                        except Exception as transform_e:

                            print(f"[CRITICAL WARNING] 坐标变换应用失败: {transform_e}")

                            import traceback

                            traceback.print_exc()

                            # 使用原始坐标继续

                    

                    # 优化网格数据处理，提高性能

                    # 确保顶点和三角形数据的精�?

                    vertices = np.array(vertices, dtype=np.float64)  # 使用高精度浮点数

                    triangles = np.array(triangles, dtype=np.int32)

                    

                    # 创建PyVista所需的面格式 - 使用更高效的实现

                    faces = np.zeros((len(triangles), 4), dtype=np.int32)

                    faces[:, 0] = 3  # 每个面有3个顶点

                    faces[:, 1:] = triangles  # 三个顶点索引

                    

                    # 创建PolyData - 直接传递数据，避免中间操作

                    pv_mesh = pv.PolyData(vertices, faces)

                    print("[CRITICAL DEBUG] PyVista网格创建成功")

                    

                    # 优化网格：清理和修复

                    pv_mesh = pv_mesh.clean(tolerance=1e-6)  # 清理重复点和退化面

                    print("[CRITICAL DEBUG] PyVista网格清理完成")

                    

                    # 优化网格：平滑处理，提高模型显示质量

                    if hasattr(pv_mesh, 'n_points') and pv_mesh.n_points < 100000:  # 只对较小模型平滑

                        pv_mesh = pv_mesh.smooth(n_iter=5, relaxation_factor=0.1)  # 轻度平滑，保留细�?

                        print("[CRITICAL DEBUG] PyVista网格平滑完成")

                    

                except Exception as inner_e:

                    print(f"[CRITICAL ERROR] 处理Open3D网格属性时出错: {inner_e}")

                    import traceback

                    traceback.print_exc()

                    return False

            else:

                # 假设已经是PyVista格式

                print(f"[CRITICAL DEBUG] 假设输入是PyVista格式")

                pv_mesh = mesh

            

            # 验证PyVista网格

            if pv_mesh is None:

                print(f"[CRITICAL ERROR] 网格转换失败，pv_mesh为None")

                return False

            

            # 检查网格有效�?

            if not hasattr(pv_mesh, 'n_points') or pv_mesh.n_points == 0:

                print(f"[CRITICAL ERROR] 转换后的网格无效 - 没有点数量")

                return False

            

            # 保存原始模型（未简化的模型

            self.original_models[model_name] = pv_mesh.copy()

            

            # 自动简化大型模型以提高性能，但保留更多细节

            if hasattr(pv_mesh, 'n_points') and pv_mesh.n_points > 100000:  # 提高简化阈值，只对超大型模型简�?

                print(f"[CRITICAL DEBUG] 检测到大型模型，点�?{pv_mesh.n_points}，开始简�?..")

                try:

                    # 使用PyVista的简化功能，保留更多面（70%�?

                    reduction_factor = min(0.3, 100000 / pv_mesh.n_points)  # 控制简化比例，保留更多细节

                    pv_mesh = pv_mesh.decimate(reduction_factor)

                    print(f"[CRITICAL DEBUG] 模型简化完成，新点�?{pv_mesh.n_points}")

                except Exception as decimate_e:

                    print(f"[CRITICAL WARNING] 模型简化失败 {decimate_e}")

                    # 继续使用原始模型，不影响后续流程

            

            # 合并日志输出，减少I/O操作

            print(f"[CRITICAL DEBUG] 转换后网�? 点数={pv_mesh.n_points}, 单元�?{pv_mesh.n_cells if hasattr(pv_mesh, 'n_cells') else '未知'}")

            

            # 移除已存在的同名模型

            if model_name in self.model_actors:

                print(f"[CRITICAL DEBUG] 移除已存在的同名模型: {model_name}")

                try:

                    self.plotter.remove_actor(self.model_actors[model_name])

                    del self.model_actors[model_name]

                except Exception as remove_e:

                    print(f"[CRITICAL ERROR] 移除现有模型失败: {remove_e}")

            

            # 使用优化的渲染参数添加网�?

            print("[CRITICAL DEBUG] 添加网格到查看器...")

            try:

                # 优化渲染参数，增强模型的立体感和清晰度

                actor = self.plotter.add_mesh(

                    pv_mesh,

                    color=color,

                    opacity=1.0 - self.current_transparency,  # 根据透明度设�?

                    show_edges=False,          # 禁用边缘显示以提高性能

                    name=model_name,

                    specular=0.5,              # 增加反光，增强立体感

                    ambient=0.3,               # 增加环境光，提高模型亮度

                    diffuse=0.8,               # 调整漫反射，使颜色更均匀

                    pickable=True,             # 确保模型可拾�?

                    smooth_shading=True        # 启用平滑着色，提高模型质感

                )

                print(f"[CRITICAL DEBUG] 网格添加成功，actor: {actor}")

                

                # 保存actor引用

                self.model_actors[model_name] = actor

                

            except Exception as add_mesh_e:

                print(f"[CRITICAL ERROR] 添加网格到场景失败 {add_mesh_e}")

                import traceback

                traceback.print_exc()

                return False

            

            # 重置相机并设置视图- 减少日志输出
            try:
                # 确保plotter仍然存在
                if self.plotter is None:
                    print(f"[CRITICAL ERROR] 查看器已不存在，无法设置视图")
                    return False
                    
                self.plotter.reset_camera()
                self.plotter.view_xy()
                # 强制渲染
                self.plotter.render()
                print("[CRITICAL DEBUG] 相机重置和视图设置成功")
            except Exception as camera_e:
                print(f"[CRITICAL WARNING] 相机操作失败: {camera_e}")
                # 尝试延迟执行相机操作
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(50, lambda: self.plotter.reset_camera() if self.plotter else None)
                QTimer.singleShot(100, lambda: self.plotter.view_xy() if self.plotter else None)

            

            # 发送成功信�?

            self.model_loaded.emit(model_name)

            print("[CRITICAL DEBUG] ====== 模型添加成功 ======\n")

            return True

        except Exception as e:
            error_msg = f"添加模型 {model_name} 失败: {str(e)}"
            print(f"[CRITICAL ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # 确保在异常情况下清理资源
            try:
                if hasattr(self, 'models') and model_name in self.models:
                    del self.models[model_name]
                if hasattr(self, 'model_actors') and model_name in self.model_actors:
                    actor = self.model_actors[model_name]
                    if hasattr(self, 'plotter') and self.plotter and actor:
                        try:
                            self.plotter.remove_actor(actor)
                        except:
                            pass
                    del self.model_actors[model_name]
                if hasattr(self, 'original_models') and model_name in self.original_models:
                    del self.original_models[model_name]
            except Exception as cleanup_e:
                print(f"[CRITICAL WARNING] 清理资源时出错: {cleanup_e}")
                
            self.model_error.emit(error_msg)
            return False

    

    def show_contact_points(self, contact_points, point_size=1.0):

        """显示接触面特征点



        Args:

            contact_points: 接触面特征点数组，形状为(N, 3)

            point_size: 点的大小



        Returns:

            bool: 是否显示成功

        """

        print("[MODEL_VIEWER] 开始显示接触面特征�?..")



        try:

            # 验证输入

            if contact_points is None:

                print("[CRITICAL ERROR] 接触面特征点为None")

                return False



            contact_points = np.array(contact_points)

            if contact_points.ndim != 2 or contact_points.shape[1] != 3:

                print(f"[CRITICAL ERROR] 接触面特征点形状无效: {contact_points.shape}")

                return False

                

            if len(contact_points) == 0:

                print("[CRITICAL ERROR] 接触面特征点数量�?")

                return False

                

            # 移除已存在的接触面特征点

            try:

                if hasattr(self, 'contact_points_actor') and self.contact_points_actor:

                    print("[CRITICAL DEBUG] 移除已存在的接触面特征点")

                    self.plotter.remove_actor(self.contact_points_actor)

                    self.contact_points_actor = None

            except Exception as remove_e:

                print(f"[CRITICAL WARNING] 移除现有接触面特征点失败: {remove_e}")

            

            # 创建点云数据

            try:

                # 创建点集

                points_mesh = pv.PolyData(contact_points)

                print(f"[CRITICAL DEBUG] 成功创建接触面特征点点云，点�? {len(contact_points)}")

                

                # 使用黄色标记接触面特征点

                self.contact_points_actor = self.plotter.add_mesh(

                    points_mesh,

                    color=[1, 1, 0],  # 黄色

                    point_size=point_size,

                    render_points_as_spheres=True,

                    opacity=0.8,

                    name="contact_points",

                    reset_camera=False

                )

                print("[CRITICAL DEBUG] 接触面特征点添加成功")

                

                # 渲染更新

                try:

                    self.plotter.render()

                    print("[CRITICAL DEBUG] 渲染更新完成")

                except Exception as render_e:

                    print(f"[CRITICAL WARNING] 渲染失败: {render_e}")

                    

                return True

                

            except Exception as create_points_e:

                print(f"[CRITICAL ERROR] 创建接触面特征点点云失败: {create_points_e}")

                import traceback

                traceback.print_exc()

                return False

                

        except Exception as e:

            print(f"[CRITICAL ERROR] 显示接触面特征点时发生错�? {str(e)}")

            import traceback

            traceback.print_exc()

            return False

    



    



    



    



    



    



    

    def remove_model(self, model_name):

        """移除指定的模�?

        

        Args:

            model_name: 要移除的模型名称

        """

        if model_name in self.models:

            # 移除模型

            del self.models[model_name]

            

            # 移除对应的actor

            if model_name in self.model_actors:

                actor = self.model_actors[model_name]

                self.plotter.remove_actor(actor)

                del self.model_actors[model_name]

            

            # 如果没有模型了，重置数据

            if not self.models:

                self.reset_data()

            

            # 更新视图

            self.plotter.render()

    

    def enable_marking(self, mode="plane"):

        """启用标记功能，连接鼠标事�?

        

        Args:

            mode: 标记模式�?plane"表示标记平面�?maxilla"表示标记上颌�?incisive_papilla"表示标记切牙乳突前

                 "mandible_crest"表示标记下颌后槽牙槽嵴，"alveolar_ridge"表示标记牙槽嵴位置

        """

        print(f"[MODEL_VIEWER] 启用标记功能，模�? {mode}")

        

        # 保存当前标记点和平面状态，确保不会丢失

        # 初始化必要的属性（如果不存在）

        if not hasattr(self, 'marked_points'):

            self.marked_points = []

        if not hasattr(self, 'marked_points_modes'):

            self.marked_points_modes = []

        if not hasattr(self, 'plane_fitted'):

            self.plane_fitted = False

        if not hasattr(self, 'plane_params'):

            self.plane_params = None

        if not hasattr(self, 'plane_actor'):

            self.plane_actor = None

        if not hasattr(self, 'marked_points_actor'):

            self.marked_points_actor = None

        if not hasattr(self, 'marked_lines_actor'):

            self.marked_lines_actor = None

        

        # 初始化切牙乳突相关属性（如果不存在）

        if not hasattr(self, 'incisive_papilla_point'):

            self.incisive_papilla_point = None

        if not hasattr(self, 'incisive_papilla_anterior_point'):

            self.incisive_papilla_anterior_point = None

        if not hasattr(self, 'symmetry_line_actor'):

            self.symmetry_line_actor = None

        if not hasattr(self, 'cutting_plane_actor'):

            self.cutting_plane_actor = None

        

        # 初始化下颌后槽牙槽嵴相关属性（如果不存在）

        if not hasattr(self, 'mandible_crest_points'):

            self.mandible_crest_points = []

        if not hasattr(self, 'mandible_crest_line_actor'):

            self.mandible_crest_line_actor = None

        if not hasattr(self, 'mandible_crest_projection_actor'):

            self.mandible_crest_projection_actor = None

        

        # 初始化牙槽嵴相关属性（如果不存在）
        if not hasattr(self, 'alveolar_ridge_points'):
            self.alveolar_ridge_points = []
        if not hasattr(self, 'alveolar_ridge_curve_actor'):
            self.alveolar_ridge_curve_actor = None
        
        # 初始化划分上颌相关属性（如果不存在）
        if not hasattr(self, 'divide_maxilla_points'):
            self.divide_maxilla_points = []
        if not hasattr(self, 'divide_maxilla_line_actor'):
            self.divide_maxilla_line_actor = None
        if not hasattr(self, 'divide_maxilla_projection_actor'):
            self.divide_maxilla_projection_actor = None
        
        # 初始化划分下颌相关属性（如果不存在）
        if not hasattr(self, 'divide_mandible_points'):
            self.divide_mandible_points = []
        if not hasattr(self, 'divide_mandible_line_actor'):
            self.divide_mandible_line_actor = None
        if not hasattr(self, 'divide_mandible_projection_actor'):
            self.divide_mandible_projection_actor = None

        



        

        # 检查现有状态

        has_existing_points = len(self.marked_points) > 0

        has_fitted_plane = self.plane_fitted and self.plane_params is not None

        

        # 连接鼠标释放事件 - 使用Qt的事件过滤机�?

        self.plotter.enable_trackball_style()

        

        # 先移除可能存在的事件过滤器，避免重复安装

        try:

            self.plotter.interactor.removeEventFilter(self)

        except Exception:

            pass  # 如果过滤器不存在，忽略异�?

        

        # 安装事件过滤器来捕获鼠标事件

        self.plotter.interactor.installEventFilter(self)

        

        # 设置标记功能状态和模式

        self._marking_enabled = True

        self._marking_mode = mode

        

        # 清理现有的标记点和平面显示，为重新显示做准备

        if self.marked_points_actor:

            try:

                self.plotter.remove_actor(self.marked_points_actor)

                self.marked_points_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除标记点演员时出错: {e}")

        

        if self.plane_actor:

            try:

                self.plotter.remove_actor(self.plane_actor)

                self.plane_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除平面演员时出�? {e}")

        

        if hasattr(self, 'marked_lines_actor') and self.marked_lines_actor:

            try:

                self.plotter.remove_actor(self.marked_lines_actor)

                self.marked_lines_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除标记线演员时出错: {e}")

        

        # 清理切牙乳突相关显示

        if hasattr(self, 'symmetry_line_actor') and self.symmetry_line_actor:

            try:

                self.plotter.remove_actor(self.symmetry_line_actor)

                self.symmetry_line_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除对称线演员时出错: {e}")

        

        if hasattr(self, 'cutting_plane_actor') and self.cutting_plane_actor:

            try:

                self.plotter.remove_actor(self.cutting_plane_actor)

                self.cutting_plane_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除切割平面演员时出�? {e}")

        

        # 清理下颌后槽牙槽嵴相关显示和数据

        if hasattr(self, 'mandible_crest_line_actor') and self.mandible_crest_line_actor:

            try:

                self.plotter.remove_actor(self.mandible_crest_line_actor)

                self.mandible_crest_line_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除下颌后槽牙槽嵴连线演员时出错: {e}")

        

        if hasattr(self, 'mandible_crest_projection_actor') and self.mandible_crest_projection_actor:

            try:

                self.plotter.remove_actor(self.mandible_crest_projection_actor)

                self.mandible_crest_projection_actor = None

            except Exception as e:

                print(f"[MODEL_VIEWER] 移除下颌后槽牙槽嵴投影演员时出错: {e}")

        

        # 当切换到特定模式时，只清空对应模式的旧标记点
        if mode == "mandible_crest":
            self.mandible_crest_points = []
        elif mode == "alveolar_ridge":
            self.alveolar_ridge_points = []
        elif mode == "divide_maxilla":
            self.divide_maxilla_points = []

        

        # 重新显示所有标记点，包括标准标记点、牙槽嵴标记点和下颌后槽牙槽嵴标记点

        print("[MODEL_VIEWER] 重新显示所有标记点")

        self._show_marked_points()

        

        # 发送标记点更新信号，确保UI同步

        if has_existing_points:

            self.marked_points_updated.emit(self.marked_points)

        elif hasattr(self, 'alveolar_ridge_points') and self.alveolar_ridge_points:

            self.marked_points_updated.emit(self.alveolar_ridge_points)

        elif hasattr(self, 'mandible_crest_points') and self.mandible_crest_points:

            self.marked_points_updated.emit(self.mandible_crest_points)

        

        # 如果存在拟合平面，重新显示它

        if has_fitted_plane:

            print("[MODEL_VIEWER] 保留现有拟合平面并重新显示")

            self.show_plane()

        

        # 显示模式特定的提示信�?

        if mode == "plane":

            print("[MODEL_VIEWER] 标记平面功能已启用，右键点击空间内任意位置标记点")

            print("[MODEL_VIEWER] 提示: Plane模式下，三点即可拟合平面")

            print("[MODEL_VIEWER] 提示: 平面拟合完成后将在多次操作间保持存在")

        elif mode == "maxilla":

            print("[MODEL_VIEWER] 标记上颌功能已启用，右键点击上颌模型表面标记")

            print("[MODEL_VIEWER] 提示: Maxilla模式下，建议在关键位置标记多个点以提高精度")

            print("[MODEL_VIEWER] 提示: 标记点和平面将在多次操作间保持存在")

        elif mode == "incisive_papilla":
            print("[MODEL_VIEWER] 标记切牙乳突功能已启用")
            print("[MODEL_VIEWER] 提示: 请先标记切牙乳突位置")

            print("[MODEL_VIEWER] 提示: 然后标记切牙乳突前-10mm的位置")

        elif mode == "mandible_crest":

            print("[MODEL_VIEWER] 标记下颌后槽牙槽嵴功能已启用")

            print("[MODEL_VIEWER] 提示: 请标记下颌后槽牙槽嵴左右两侧的位置")

            print("[MODEL_VIEWER] 提示: 标记完成后将自动显示连线并投影到𬌗平面")

        elif mode == "alveolar_ridge":
            print("[MODEL_VIEWER] 标记牙槽嵴位置功能已启用")
            print("[MODEL_VIEWER] 提示: 请在牙槽嵴上标记多个点")
            print("[MODEL_VIEWER] 提示: 标记完成后将自动拟合形成曲线")
            print("[MODEL_VIEWER] 提示: 可用于上颌或下颌模型")
        elif mode == "divide_maxilla":
            print("[MODEL_VIEWER] 划分上颌功能已启用")
            print("[MODEL_VIEWER] 提示: 请在模型上标记点，标记点之间将用直线连接")
            print("[MODEL_VIEWER] 提示: 标记完成后将自动显示连接线并投影到牙合平面")

        

        # 刷新视图确保所有更改都生效

        try:

            self.plotter.render()

        except Exception as e:

            print(f"[MODEL_VIEWER] 渲染视图时出�? {e}")

        

        return True

        

    def eventFilter(self, obj, event):

        """Qt事件过滤器，用于捕获鼠标事件"""

        # 检查是否启用了标记功能

        if not hasattr(self, '_marking_enabled') or not self._marking_enabled:

            return False

            

        # 检查是否是右键释放事件

        if event.type() == event.MouseButtonRelease and event.button() == Qt.RightButton:

            # 添加调试信息

            print(f"[DEBUG] Qt右键释放事件触发")

            print(f"[DEBUG] 获取到的屏幕坐标: x={event.x()}, y={event.y()}")

            

            # 调用原有的right_click_callback方法处理标记点添�?

            class DummyEvent:

                def __init__(self, position):

                    self.position = position

            

            self.right_click_callback(DummyEvent((event.x(), event.y())))

            return True

        

        return False

    

    def right_click_callback(self, event):

        """处理鼠标释放事件，标记点"""

        # 获取屏幕坐标

        x, y = event.position

        

        # 使用PyVista的光线投射功能获取模型表面点坐标（类似Meshlab的pick points功能�?

        try:

            # 获取窗口尺寸

            size = self.plotter.size()

            width = size.width()

            height = size.height()

            

            # 调试信息

            print(f"[MODEL_VIEWER] 右键点击屏幕坐标: ({x}, {y})")

            print(f"[MODEL_VIEWER] 窗口尺寸: {width}x{height}")

            

            # 将屏幕坐标转换为归一化设备坐�?(NDC)

            # 注意：PyVista的屏幕坐标原点是左上角，Y轴向量

            ndc_x = (2.0 * x / width) - 1.0

            ndc_y = 1.0 - (2.0 * y / height)  # 转换Y轴方向，使原点在左下�?

            

            print(f"[MODEL_VIEWER] 归一化设备坐�? ({ndc_x:.4f}, {ndc_y:.4f})")

            

            # 获取相机参数

            camera = self.plotter.camera

            if camera is None:

                camera = self.plotter.renderer.camera

            

            camera_pos = np.array(camera.GetPosition())

            camera_focal = np.array(camera.GetFocalPoint())

            camera_view_up = np.array(camera.GetViewUp())

            

            print(f"[MODEL_VIEWER] 相机位置: {camera_pos}")

            print(f"[MODEL_VIEWER] 相机焦点: {camera_focal}")

            print(f"[MODEL_VIEWER] 相机上向量 {camera_view_up}")

            

            # 计算相机的方向向量

            camera_dir = camera_focal - camera_pos

            camera_dir_norm = np.linalg.norm(camera_dir)

            if camera_dir_norm < 1e-6:

                print("[MODEL_VIEWER ERROR] 相机方向向量长度接近零")

                return

            camera_dir /= camera_dir_norm

            

            # 计算相机的右向量

            camera_right = np.cross(camera_dir, camera_view_up)

            camera_right_norm = np.linalg.norm(camera_right)

            if camera_right_norm < 1e-6:

                print("[MODEL_VIEWER ERROR] 相机右向量长度接近零")

                return

            camera_right /= camera_right_norm

            

            # 计算相机的上向量

            camera_up = np.cross(camera_right, camera_dir)

            camera_up_norm = np.linalg.norm(camera_up)

            if camera_up_norm < 1e-6:

                print("[MODEL_VIEWER ERROR] 相机上向量长度接近零")

                return

            camera_up /= camera_up_norm

            

            # 获取相机的视场角和纵横比

            fov_y = camera.GetViewAngle()

            aspect_ratio = width / height

            

            print(f"[MODEL_VIEWER] 视场�? {fov_y}�? 纵横�? {aspect_ratio:.2f}")

            

            # 计算光线的方向向量

            tan_half_fov = np.tan(np.radians(fov_y) / 2.0)

            ray_dir = (camera_dir +

                      ndc_x * camera_right * tan_half_fov * aspect_ratio +

                      ndc_y * camera_up * tan_half_fov)

            ray_dir_norm = np.linalg.norm(ray_dir)

            if ray_dir_norm < 1e-6:

                print("[MODEL_VIEWER ERROR] 光线方向向量长度接近零")

                return

            ray_dir /= ray_dir_norm

            

            print(f"[MODEL_VIEWER] 光线方向: {ray_dir}")

            

            # 使用相机位置作为光线起点

            ray_start = camera_pos

            

            # 定义光线终点（足够远，确保能击中模型�?

            ray_end = ray_start + ray_dir * 1000.0  # 增加光线长度�?000

            

            # 遍历所有模型，查找光线与模型的交点

            point = None

            closest_distance = float('inf')

            

            # 调试：打印可用模型信�?

            print(f"[MODEL_VIEWER] original_models 数量: {len(self.original_models) if hasattr(self, 'original_models') else 0}")

            print(f"[MODEL_VIEWER] models 数量: {len(self.models) if hasattr(self, 'models') else 0}")

            

            # 优先使用original_models中的转换后的PyVista网格（已应用变换�?

            if hasattr(self, 'original_models') and self.original_models:

                print("[MODEL_VIEWER] 开始在转换后的模型上进行光线追�?..")

                for model_name, pv_mesh in self.original_models.items():

                    try:

                        # 检查模型是否可见

                        if model_name not in self.model_actors:

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 不在演员列表中，跳过光线追踪")

                            continue

                        

                        actor = self.model_actors[model_name]

                        if not actor.GetVisibility():

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 不可见，跳过光线追踪")

                            continue

                        

                        # 检查模型是否有�?

                        if not isinstance(pv_mesh, pv.DataSet):

                            print(f"[MODEL_VIEWER WARNING] 模型 '{model_name}' 不是有效的PyVista数据集")

                            continue

                        

                        if pv_mesh.n_points == 0:

                            print(f"[MODEL_VIEWER WARNING] 模型 '{model_name}' 没有顶点")

                            continue

                        

                        if pv_mesh.n_cells == 0:

                            print(f"[MODEL_VIEWER WARNING] 模型 '{model_name}' 没有单元")

                            continue

                        

                        print(f"[MODEL_VIEWER] 检查模�?'{model_name}': 顶点�?{pv_mesh.n_points}, 单元�?{pv_mesh.n_cells}")

                        

                        # 使用ray_trace方法检测交�?

                        intersections, _ = pv_mesh.ray_trace(ray_start, ray_end, first_point=True)

                        

                        if intersections is not None and len(intersections) > 0:

                            # 计算交点距离

                            dist = np.linalg.norm(intersections - ray_start)

                            if dist < closest_distance:

                                point = intersections

                                closest_distance = dist

                                print(f"[MODEL_VIEWER] 成功拾取模型 '{model_name}' 表面�? {point}, 距离: {dist:.4f}")

                    except Exception as e:

                        print(f"[MODEL_VIEWER WARNING] 对模�?'{model_name}' 进行光线追踪失败: {e}")

                        import traceback

                        traceback.print_exc()

            

            # 如果在original_models中没有找到交点，尝试使用models中的网格

            if point is None and hasattr(self, 'models') and self.models:

                print("[MODEL_VIEWER] 开始在原始模型上进行光线追�?..")

                for model_name, model_mesh in self.models.items():

                    try:

                        # 检查模型是否可见

                        if model_name not in self.model_actors:

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 不在演员列表中，跳过光线追踪")

                            continue

                        

                        actor = self.model_actors[model_name]

                        if not actor.GetVisibility():

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 不可见，跳过光线追踪")

                            continue

                        

                        pv_mesh = None

                        

                        # 检查模型类型并转换为PyVista网格

                        if isinstance(model_mesh, pv.PolyData):

                            pv_mesh = model_mesh

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 是PyVista网格")

                        elif isinstance(model_mesh, o3d.geometry.TriangleMesh):

                            # 将Open3D网格转换为PyVista网格用于光线追踪

                            vertices = np.asarray(model_mesh.vertices)

                            triangles = np.asarray(model_mesh.triangles)

                            

                            if vertices.size == 0 or triangles.size == 0:

                                print(f"[MODEL_VIEWER WARNING] 模型 '{model_name}' 没有顶点或三角形")

                                continue

                                

                            faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])

                            pv_mesh = pv.PolyData(vertices, faces)

                            print(f"[MODEL_VIEWER] 模型 '{model_name}' 转换为PyVista网格: 顶点�?{vertices.shape[0]}, 三角形数={triangles.shape[0]}")

                        else:

                            print(f"[MODEL_VIEWER WARNING] 模型 '{model_name}' 类型不支�? {type(model_mesh)}")

                            continue

                        

                        if pv_mesh is not None and pv_mesh.n_points > 0 and pv_mesh.n_cells > 0:

                            # 使用ray_trace方法检测交�?

                            intersections, _ = pv_mesh.ray_trace(ray_start, ray_end, first_point=True)

                            

                            if intersections is not None and len(intersections) > 0:

                                # 计算交点距离

                                dist = np.linalg.norm(intersections - ray_start)

                                if dist < closest_distance:

                                    point = intersections

                                    closest_distance = dist

                                    print(f"[MODEL_VIEWER] 成功从原始模�?'{model_name}' 拾取表面�? {point}, 距离: {dist:.4f}")

                    except Exception as e:

                        print(f"[MODEL_VIEWER WARNING] 对原始模�?'{model_name}' 进行光线追踪失败: {e}")

                        import traceback

                        traceback.print_exc()

            

            # 如果没有击中任何模型，使用相机前方的�?

            if point is None:

                # 使用相机前方50个单位的点（更合理的默认值）

                point = ray_start + ray_dir * 50.0

                print(f"[MODEL_VIEWER] 未击中任何模型，使用相机前方向 {point}")

            

            # 添加标记�?

            self.add_marked_point(point)

                

        except Exception as e:

            print(f"[MODEL_VIEWER ERROR] 空间坐标转换失败: {e}")

            import traceback

            traceback.print_exc()

    

    def _check_point_limit(self):

        """检查当前模式下是否已达到最大标记点数限�?

        

        Returns:

            bool: True表示已达到限制，False表示未达到限�?

        """

        if not hasattr(self, '_marking_mode'):

            return False

        

        if self._marking_mode == "plane" and len(self.marked_points) >= 3:

            print("[MODEL_VIEWER] 平面标记模式下已达到最大标记点数（3个）")

            return True

        elif self._marking_mode == "incisive_papilla":

            # 切牙乳突模式下，最多标�?个点

            ip_points_count = sum(1 for mode in self.marked_points_modes if mode == 'incisive_papilla') if hasattr(self, 'marked_points_modes') else 0

            if ip_points_count >= 2:

                print("[MODEL_VIEWER] 切牙乳突标记模式下已达到最大标记点数（2个）")

                return True

        elif self._marking_mode == "mandible_crest":

            # 下颌后槽牙槽嵴模式下，最多标�?个点（左右两侧）

            if hasattr(self, 'mandible_crest_points') and len(self.mandible_crest_points) >= 2:

                print("[MODEL_VIEWER] 下颌后槽牙槽嵴标记模式下已达到最大标记点数（2个）")

                return True

        

        return False

        

    def _add_mandible_crest_point(self, point):

        """添加下颌后槽牙槽嵴标记点

        

        Args:

            point: 3D点坐�?

        """

        # 确保点列表存�?

        if not hasattr(self, 'mandible_crest_points'):

            self.mandible_crest_points = []

        

        # 添加点并更新状态

        self.mandible_crest_points.append(point)

        print(f"[MODEL_VIEWER] 添加下颌后槽牙槽嵴标记点: {point}")

        print(f"[MODEL_VIEWER] 当前下颌后槽牙槽嵴标记点�? {len(self.mandible_crest_points)}")

        

        # 立即显示刚添加的标记�?

        self._show_mandible_crest_points()

        

        # 如果�?个点，显示连接线并投影到𬌗平面

        if len(self.mandible_crest_points) == 2:

            self.show_mandible_crest_line()

            self.project_mandible_crest_to_plane()

        

        # 发送标记点更新信号

        self.marked_points_updated.emit(self.mandible_crest_points)

        

    def _add_alveolar_ridge_point(self, point):

        """添加牙槽嵴标记点

        

        Args:

            point: 3D点坐�?

        """

        # 确保点列表存�?

        if not hasattr(self, 'alveolar_ridge_points'):

            self.alveolar_ridge_points = []

        

        # 添加点并更新状态

        self.alveolar_ridge_points.append(point)

        print(f"[MODEL_VIEWER] 添加牙槽嵴标记点: {point}")

        print(f"[MODEL_VIEWER] 当前牙槽嵴标记点�? {len(self.alveolar_ridge_points)}")

        

        # 更新显示

        self._show_marked_points()

        

        # 发送标记点更新信号
        self.marked_points_updated.emit(self.alveolar_ridge_points)
    
    def _add_divide_maxilla_point(self, point):
        """添加划分上颌标记点
        
        Args:
            point: 3D点坐�?
        """
        # 确保点列表存�?
        if not hasattr(self, 'divide_maxilla_points'):
            self.divide_maxilla_points = []
        
        # 添加点并更新状态
        self.divide_maxilla_points.append(point)
        print(f"[MODEL_VIEWER] 添加划分上颌标记点: {point}")
        print(f"[MODEL_VIEWER] 当前划分上颌标记点�? {len(self.divide_maxilla_points)}")
        
        # 更新显示
        self._show_marked_points()
        
        # 当有两个或更多点时，显示连接线并投影到牙合平面
        if len(self.divide_maxilla_points) >= 2:
            self.show_divide_maxilla_line()
            self.project_divide_maxilla_to_plane()
        
        # 发送标记点更新信号
        self.marked_points_updated.emit(self.divide_maxilla_points)
    
    def _add_divide_mandible_point(self, point):
        """添加划分下颌标记点
        
        Args:
            point: 3D点坐�?
        """
        # 确保点列表存�?
        if not hasattr(self, 'divide_mandible_points'):
            self.divide_mandible_points = []
        
        # 添加点并更新状态
        self.divide_mandible_points.append(point)
        print(f"[MODEL_VIEWER] 添加划分下颌标记点: {point}")
        print(f"[MODEL_VIEWER] 当前划分下颌标记点�? {len(self.divide_mandible_points)}")
        
        # 更新显示
        self._show_marked_points()
        
        # 当有两个或更多点时，显示连接线并投影到牙合平面
        if len(self.divide_mandible_points) >= 2:
            self.show_divide_mandible_line()
            self.project_divide_mandible_to_plane()
        
        # 发送标记点更新信号
        self.marked_points_updated.emit(self.divide_mandible_points)
    
    def add_marked_point(self, point):

        """添加标记�?
        
        Args:
            point: 3D点坐�?
        """
        # 检查标记点数量限制
        if self._check_point_limit():
            return
        
        # 处理下颌后槽牙槽嵴、牙槽嵴、划分上颌、划分下颌模式的特殊逻辑
        if hasattr(self, '_marking_mode'):
            if self._marking_mode == "mandible_crest":
                self._add_mandible_crest_point(point)
                return
            elif self._marking_mode == "alveolar_ridge":
                self._add_alveolar_ridge_point(point)
                return
            elif self._marking_mode == "divide_maxilla":
                self._add_divide_maxilla_point(point)
                return
            elif self._marking_mode == "divide_mandible":
                self._add_divide_mandible_point(point)
                return


        
        # 处理其他模式的标准逻辑
        # 添加标记点并记录其来源模�?
        self.marked_points.append(point)
        
        # 初始化或更新标记点模式记录
        if not hasattr(self, 'marked_points_modes'):
            self.marked_points_modes = []
        current_mode = self._marking_mode if hasattr(self, '_marking_mode') else 'plane'
        self.marked_points_modes.append(current_mode)
        
        print(f"[MODEL_VIEWER] 添加标记�? {point}")
        print(f"[MODEL_VIEWER] 当前标记点数: {len(self.marked_points)}")
        print(f"[MODEL_VIEWER] 当前标记模式: {current_mode}")
        print(f"[MODEL_VIEWER] 点来源模式记�? {self.marked_points_modes}")
        
        # 显示标记�?
        self._show_marked_points()

        

        # 处理切牙乳突模式的特殊逻辑

        if hasattr(self, '_marking_mode') and self._marking_mode == "incisive_papilla":

            # 获取所有切牙乳突模式的�?

            ip_points = [p for i, p in enumerate(self.marked_points) if self.marked_points_modes[i] == 'incisive_papilla']

            

            if len(ip_points) == 1:

                # 第一个点是切牙乳突点

                self.incisive_papilla_point = ip_points[0]

                print(f"[MODEL_VIEWER] 已标记切牙乳突点: {self.incisive_papilla_point}")

            elif len(ip_points) == 2:

                # 第二个点是切牙乳突前8-10mm位置�?

                self.incisive_papilla_anterior_point = ip_points[1]

                print(f"[MODEL_VIEWER] 已标记切牙乳突前位置�? {self.incisive_papilla_anterior_point}")

                # 计算并显示对称线

                self.calculate_incisive_papilla_anterior()

        

        # 生成标记线：按顺序连接当前模式的所有标记点

        if hasattr(self, '_marking_mode'):

            current_mode = self._marking_mode

            # 收集当前模式的所有点

            current_mode_points = [p for i, p in enumerate(self.marked_points) if self.marked_points_modes[i] == current_mode]

            

            # 重新生成当前模式的所有连�?

            # 先移除当前模式的旧连�?

            self.marker_lines = [line for line in self.marker_lines if line[2] != current_mode]

            

            # 添加新的连线

            if len(current_mode_points) >= 2:

                for i in range(len(current_mode_points) - 1):

                    p1 = current_mode_points[i]

                    p2 = current_mode_points[i + 1]

                    # 存储连线及其模式�?p1, p2, mode)

                    self.marker_lines.append((p1, p2, current_mode))

            

            print(f"[MODEL_VIEWER] 当前模式 {current_mode} 已生�?{len(self.marker_lines)} 条标记线")

        

        # 发送标记点更新信号

        self.marked_points_updated.emit(self.marked_points)

        

        # 只有在标记平面模式下，当标记�?个点时才自动拟合平面

        if hasattr(self, '_marking_mode') and self._marking_mode == "plane" and len(self.marked_points) == 3:

            self.fit_plane_from_points()

        # 在maxilla模式下，当添加点时生成垂直平面

        elif hasattr(self, '_marking_mode') and self._marking_mode == "maxilla":

            self.generate_maxilla_vertical_plane()

    

    def _show_marked_points(self):

        """显示标记点，并提供用户友好的提示信息"""

        # 移除已存在的标记�?

        if self.marked_points_actor:

            self.plotter.remove_actor(self.marked_points_actor)

            self.marked_points_actor = None

        

        # 处理不同模式的显示，显示所有标记点

        

        # 1. 首先处理标准模式标记点（plane, maxilla, incisive_papilla�?

        standard_points_shown = False

        if self.marked_points:

            standard_points_shown = True

        

        # 2. 处理牙槽嵴模式标记点

        alveolar_points_shown = False

        if hasattr(self, '_marking_mode') and self._marking_mode == "alveolar_ridge" and hasattr(self, 'alveolar_ridge_points') and self.alveolar_ridge_points:

            alveolar_points_shown = True

        

        # 3. 处理下颌后槽牙槽嵴模式标记点

        mandible_points_shown = False

        if hasattr(self, '_marking_mode') and self._marking_mode == "mandible_crest" and hasattr(self, 'mandible_crest_points') and self.mandible_crest_points:

            mandible_points_shown = True

        

        # 4. 如果没有任何标记点，直接返回

        if not standard_points_shown and not alveolar_points_shown and not mandible_points_shown:

            return

        

        # 创建标记点点�?

        points_array = np.array(self.marked_points)

        points_mesh = pv.PolyData(points_array)

        

        # 为每个标记点单独设置颜色

        # 需要记录每个标记点的来源模�?

        if not hasattr(self, 'marked_points_modes'):

            # 初始化标记点模式记录，如果之前没有记录，则假设所有点都是平面模式

            self.marked_points_modes = ['plane'] * len(self.marked_points)

        elif len(self.marked_points_modes) != len(self.marked_points):

            # 如果点数量不匹配，调整模式记录长�?

            self.marked_points_modes = self.marked_points_modes[:len(self.marked_points)]

            if len(self.marked_points_modes) < len(self.marked_points):

                # 如果有新点，使用当前模式

                self.marked_points_modes.extend([self._marking_mode] * (len(self.marked_points) - len(self.marked_points_modes)))

        

        # 根据模式设置不同颜色

        point_colors = np.zeros((len(points_array), 3))

        for i, mode in enumerate(self.marked_points_modes):

            if mode == 'plane':

                point_colors[i] = [1.0, 0.0, 0.0]  # 红色

            elif mode == 'maxilla':

                point_colors[i] = [0.0, 0.0, 1.0]  # 蓝色

            elif mode == 'incisive_papilla':

                # 切牙乳突模式下，第一个点用绿色，第二个点用黄�?

                if i < len(self.marked_points_modes) - 1:
                    # 如果不是最后一个切牙乳突点，检查前面是否有切牙乳突前
                    ip_indices = [j for j, m in enumerate(self.marked_points_modes[:i+1]) if m == 'incisive_papilla']
                    if len(ip_indices) == 1:
                        point_colors[i] = [0.0, 1.0, 0.0]  # 第一个点（切牙乳突点）用绿色
                    elif len(ip_indices) == 2:
                        point_colors[i] = [1.0, 1.0, 0.0]  # 第二个点（前8-10mm点）用黄�?
                else:
                    # 最后一个点，检查前面是否有切牙乳突前
                    ip_indices = [j for j, m in enumerate(self.marked_points_modes) if m == 'incisive_papilla']
                    if len(ip_indices) == 1:
                        point_colors[i] = [0.0, 1.0, 0.0]  # 第一个点（切牙乳突点）用绿色
                    elif len(ip_indices) == 2:
                        point_colors[i] = [1.0, 1.0, 0.0]  # 第二个点（前8-10mm点）用黄�?

        

        # 显示标记点，使用点颜色数量

        self.marked_points_actor = self.plotter.add_mesh(

            points_mesh,

            scalars=point_colors,

            rgb=True,

            point_size=20.0,

            render_points_as_spheres=True,

            opacity=1.0,

            name="marked_points",

            reset_camera=False

        )

        

        # 显示牙槽嵴标记点

        if hasattr(self, 'alveolar_ridge_points') and self.alveolar_ridge_points:

            self._show_alveolar_ridge_points()

        

        # 显示下颌后槽牙槽嵴标记点

        if hasattr(self, 'mandible_crest_points') and self.mandible_crest_points:

            self._show_mandible_crest_points()

        



        

        # 添加用户友好的提示信�?

        if hasattr(self, '_marking_mode'):

            mode_name = "平面" if self._marking_mode == "plane" else "上颌" if self._marking_mode == "maxilla" else "切牙乳突" if self._marking_mode == "incisive_papilla" else "未知"

            print(f"[MODEL_VIEWER] 当前标记{mode_name}，已显示 {len(self.marked_points)} 个标记点")

            print(f"[MODEL_VIEWER] 提示: 标记点将在多次操作间保持存在，请放心切换功能")

        

        # 添加标记点编号和说明

        for i, point in enumerate(self.marked_points):

            label_text = f"点{i+1}"

            if hasattr(self, 'marked_points_modes') and self.marked_points_modes[i] == 'incisive_papilla':

                # 切牙乳突模式下，添加特殊标签

                ip_indices = [j for j, m in enumerate(self.marked_points_modes) if m == 'incisive_papilla']

                if len(ip_indices) > 0 and i == ip_indices[0]:

                    label_text = "切牙乳突"

                elif len(ip_indices) > 1 and i == ip_indices[1]:
                    label_text = "前-10mm"

            

            self.plotter.add_point_labels(

                [point],

                [label_text],

                font_size=12,

                show_points=False,

                name=f"label_{i+1}"

            )

        

        # 绘制标记点之间的连线

        if len(self.marked_points) >= 2:

            # 获取当前模式的点索引

            current_mode_points = [i for i, mode in enumerate(self.marked_points_modes) if mode == self._marking_mode]

            

            if len(current_mode_points) >= 2:

                # 移除已存在的连线

                if hasattr(self, 'marked_lines_actor') and self.marked_lines_actor:

                    self.plotter.remove_actor(self.marked_lines_actor)

                    self.marked_lines_actor = None

                

                # 创建连线的点和线�?

                mode_points_array = np.array([self.marked_points[i] for i in current_mode_points])

                

                # PyVista要求的线格式：每条线以线段长度开头，然后是点索引

                line_segments = []

                for i in range(len(mode_points_array) - 1):

                    line_segments.extend([2, i, i + 1])

                lines = np.array(line_segments)

                

                # 创建线网�?

                line_mesh = pv.PolyData(mode_points_array, lines=lines)

                

                # 根据标记模式选择颜色

                if self._marking_mode == 'plane':

                    color = 'red'  # 平面模式使用红色

                elif self._marking_mode == 'maxilla':

                    color = 'blue'  # 上颌模式使用蓝色

                elif self._marking_mode == 'incisive_papilla':

                    color = 'green'  # 切牙乳突模式使用绿色

                else:

                    color = 'gray'  # 默认颜色

                

                # 显示连线

                self.marked_lines_actor = self.plotter.add_mesh(

                    line_mesh,

                    color=color,

                    line_width=6.0,  # 适量加粗连线

                    opacity=1.0,

                    name="marked_lines",

                    reset_camera=False

                )

        

        # 刷新视图

        self.plotter.render()

    

    def clear_marked_points(self):

        """清除所有标记点"""

        # 移除标记�?

        if self.marked_points_actor:

            self.plotter.remove_actor(self.marked_points_actor)

            self.marked_points_actor = None

        

        # 移除标记点标�?

        for i in range(len(self.marked_points)):

            label_name = f"label_{i+1}"

            if label_name in self.plotter.actors:

                self.plotter.remove_actor(self.plotter.actors[label_name])

        

        # 移除标记点之间的连线

        if hasattr(self, 'marked_lines_actor') and self.marked_lines_actor:

            self.plotter.remove_actor(self.marked_lines_actor)

            self.marked_lines_actor = None

        

        # 移除拟合平面

        if self.plane_actor:

            self.plotter.remove_actor(self.plane_actor)

            self.plane_actor = None

        

        # 移除切牙乳突相关的可视化

        if hasattr(self, 'symmetry_line_actor') and self.symmetry_line_actor:

            self.plotter.remove_actor(self.symmetry_line_actor)

            self.symmetry_line_actor = None

        

        if hasattr(self, 'cutting_plane_actor') and self.cutting_plane_actor:

            self.plotter.remove_actor(self.cutting_plane_actor)

            self.cutting_plane_actor = None

        

        # 移除下颌后槽牙槽嵴相关的可视�?

        if hasattr(self, 'mandible_crest_line_actor') and self.mandible_crest_line_actor:

            self.plotter.remove_actor(self.mandible_crest_line_actor)

            self.mandible_crest_line_actor = None

        

        if hasattr(self, 'mandible_crest_projection_actor') and self.mandible_crest_projection_actor:

            self.plotter.remove_actor(self.mandible_crest_projection_actor)

            self.mandible_crest_projection_actor = None

        

        # 移除牙槽嵴相关的可视�?

        if hasattr(self, 'alveolar_ridge_curve_actor') and self.alveolar_ridge_curve_actor:

            self.plotter.remove_actor(self.alveolar_ridge_curve_actor)

            self.alveolar_ridge_curve_actor = None

        

        # 重置标记点数量

        self.marked_points = []

        self.marked_points_modes = []

        self.plane_params = None

        self.plane_fitted = False

        

        # 重置切牙乳突相关数据

        self.incisive_papilla_point = None

        self.incisive_papilla_anterior_point = None

        

        # 重置下颌后槽牙槽嵴相关数量

        if hasattr(self, 'mandible_crest_points'):

            self.mandible_crest_points = []

        

        # 重置牙槽嵴相关数量

        if hasattr(self, 'alveolar_ridge_points'):

            self.alveolar_ridge_points = []

        

        # 发送标记点更新信号

        self.marked_points_updated.emit(self.marked_points)

        

        # 刷新视图

        self.plotter.render()

        print("[MODEL_VIEWER] 已清除所有标记点")

    

    def fit_plane_from_points(self):
        """从标记点拟合平面，支持多个点并使用最小二乘法提高准确性"""

        if len(self.marked_points) < 3:

            print(f"[MODEL_VIEWER] 无法拟合平面：需要至少3个点，当前有{len(self.marked_points)}个点")

            return

        

        print("[MODEL_VIEWER] 开始拟合平面..")

        

        # 将标记点转换为numpy数组

        points = np.array(self.marked_points)

        

        # 如果只有3个点，使用原始方向

        if len(points) == 3:

            # 获取三个点

            p1, p2, p3 = points

            

            # 计算平面的法向量

            v1 = np.array(p2) - np.array(p1)

            v2 = np.array(p3) - np.array(p1)

            normal = np.cross(v1, v2)

            

            # 检查法向量是否有效

            norm = np.linalg.norm(normal)

            if norm < 1e-6:

                print("[MODEL_VIEWER] 三个点共线，无法拟合平面")

                return

            

            normal = normal / norm  # 单位化

            

            # 计算平面方程 ax + by + cz + d = 0

            a, b, c = normal

            d = -np.dot(normal, np.array(p1))

        else:

            # 使用最小二乘法拟合平面，提高准确性

            # 构建设计矩阵 A = [x, y, z, 1] 对于3D平面

            A = np.ones((len(points), 4))

            A[:, 0] = points[:, 0]  # x坐标

            A[:, 1] = points[:, 1]  # y坐标

            A[:, 2] = points[:, 2]  # z坐标

            

            # 求解 Ax = 0，使用SVD分解

            _, _, V = np.linalg.svd(A)

            plane_params = V[-1, :]

            

            # 平面方程：ax + by + cz + d = 0

            a, b, c, d = plane_params

            

            # 归一化平面参数

            norm = np.sqrt(a**2 + b**2 + c**2)

            if norm < 1e-6:

                print("[MODEL_VIEWER] 无法拟合有效的平面")

                return

            

            a, b, c, d = a/norm, b/norm, c/norm, d/norm

        

        # 计算平面拟合质量评分

        if len(points) > 3:

            # 计算所有点到平面的距离

            distances = np.abs(a*points[:, 0] + b*points[:, 1] + c*points[:, 2] + d)

            

            # 计算平均距离和标准差

            mean_distance = np.mean(distances)

            std_distance = np.std(distances)

            

            # 计算质量评分 (0-100)，距离越小评分越高

            # 评分公式为100 - min(90, mean_distance * 100)

            quality_score = max(0, 100 - min(90, mean_distance * 100))

            

            print(f"[MODEL_VIEWER] 平面拟合质量评分: {quality_score:.1f}/100")

            print(f"[MODEL_VIEWER] 平均点到平面距离: {mean_distance:.6f}")

            print(f"[MODEL_VIEWER] 距离标准差: {std_distance:.6f}")

            

            # 如果质量评分低于阈值，发出警告

            if quality_score < 70:

                print(f"[MODEL_VIEWER] 警告：平面拟合质量较低({quality_score:.1f}/100)，建议重新标记点")

        

        self.plane_params = (a, b, c, d)

        self.plane_fitted = True

        

        print(f"[MODEL_VIEWER] 平面方程：{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

        

        # 如果有maxilla标记点，更新垂直平面

        if hasattr(self, 'marked_points') and hasattr(self, 'marked_points_modes'):

            maxilla_points = [p for i, p in enumerate(self.marked_points) if self.marked_points_modes[i] == 'maxilla']

            if maxilla_points:

                print("[MODEL_VIEWER] 已有maxilla标记点，更新垂直平面")

                self.generate_maxilla_vertical_plane()

        

        # 显示平面

        self.show_plane()

        

        # 发送平面拟合完成信号

        self.fit_plane_completed.emit(self.plane_params)

    

    def show_plane(self):
        """显示拟合的平面"""

        if not self.plane_params:

            return

        

        # 移除已存在的平面

        if self.plane_actor:

            self.plotter.remove_actor(self.plane_actor)

            self.plane_actor = None

        

        # 获取模型的边界范�?

        all_points = []

        for model_name, model in self.models.items():

            if isinstance(model, pv.PolyData):

                all_points.extend(model.points)

            elif hasattr(model, 'vertices'):

                all_points.extend(np.asarray(model.vertices))

        

        if all_points:

            all_points = np.array(all_points)

            min_coords = np.min(all_points, axis=0)

            max_coords = np.max(all_points, axis=0)

        else:

            # 如果没有模型，使用默认范�?

            min_coords = np.array([-10, -10, -10])

            max_coords = np.array([10, 10, 10])

        

        # 扩展平面范围 - 使其刚好超过模型

        padding = 0.5  # 减小padding值，使平面更紧凑

        

        # 计算当前范围

        x_min = min_coords[0] - padding

        x_max = max_coords[0] + padding

        y_min = min_coords[1] - padding

        y_max = max_coords[1] + padding

        

        # 使用模型的实际Y范围，不强制与X范围相同

        # 这样可以避免平面过长

        x_range = (x_min, x_max)  # 保留原有宽度

        

        # 计算当前Y范围和长�?

        current_y_length = y_max - y_min

        current_y_center = (y_min + y_max) / 2

        

        # 将Y方向长度调整为当前的三分之一

        new_y_length = current_y_length / 3

        new_y_min = current_y_center - new_y_length / 2

        new_y_max = current_y_center + new_y_length / 2

        

        # 使用新的Y范围

        y_range = (new_y_min, new_y_max)

        

        # 计算中心位置

        x_center = (x_min + x_max) / 2

        y_center = current_y_center

        

        # 输出平面范围日志，方便调�?

        print(f"[MODEL_VIEWER] 平面范围 - X: {x_range}, Y: {y_range}")

        print(f"[MODEL_VIEWER] 模型边界 - X: [{min_coords[0]}, {max_coords[0]}], Y: [{min_coords[1]}, {max_coords[1]}]")

        print(f"[MODEL_VIEWER] 平面长度 - 原Y长度: {current_y_length:.2f}, 新Y长度: {new_y_length:.2f} (为原长度�?/3)")

        

        # 创建平面网格

        xx, yy = np.meshgrid(

            np.linspace(x_range[0], x_range[1], 50),

            np.linspace(y_range[0], y_range[1], 50)

        )

        

        a, b, c, d = self.plane_params

        

        # 如果c不为零，计算z坐标

        if abs(c) > 1e-6:

            zz = (-a * xx - b * yy - d) / c

        else:

            # 特殊情况处理

            zz = np.zeros_like(xx)

        

        # 创建平面网格

        plane_mesh = pv.StructuredGrid(xx, yy, zz)

        

        # 显示平面，使用半透明青色

        self.plane_actor = self.plotter.add_mesh(

            plane_mesh,

            color='cyan',

            opacity=0.5,

            name="fitted_plane",

            reset_camera=False

        )

        

        # 刷新视图

        self.plotter.render()

        print("[MODEL_VIEWER] 已显示拟合平面")

    

    def toggle_model_visibility(self, model_name, visible):

        """切换模型的可见�?

        

        Args:

            model_name: 模型名称

            visible: 是否可见

        """

        try:

            if model_name in self.model_actors:

                self.model_actors[model_name].SetVisibility(visible)

                self.plotter.render()

                print(f"模型 {model_name} {'可见' if visible else '隐藏'}")

        except Exception as e:

            print(f"切换模型 {model_name} 可见性失败 {e}")

    

    def update_transparency(self, transparency_value):

        """更新所有模型的透明�?

        

        Args:

            transparency_value: 透明度�?(0-100)

        """

        try:

            # 转换为-1范围

            self.current_transparency = transparency_value / 100.0

            

            # 更新所有模型的透明�?

            for actor in self.model_actors.values():

                actor.SetOpacity(1.0 - self.current_transparency)

            

            self.plotter.render()

            print(f"更新透明度为: {self.current_transparency}")

            

        except Exception as e:

            print(f"更新透明度失败 {e}")

    

    def clear_all(self):
        """清除所有模型和平面（安全增强版）"""

        print("[CRITICAL DEBUG] 开始安全清除所有模型和平面...")

        try:

            # 逐个移除模型actor，而不是清除所有actor

            print(f"[CRITICAL DEBUG] 移除模型actors，当前数量 {len(self.model_actors)}")

            for actor_name, actor in list(self.model_actors.items()):

                try:

                    self.plotter.remove_actor(actor)

                    print(f"[CRITICAL DEBUG] 移除模型actor: {actor_name}")

                except Exception as actor_e:

                    print(f"[CRITICAL WARNING] 移除模型actor {actor_name} 失败: {actor_e}")

            

            # 清空模型列表

            self.model_actors.clear()

            self.models.clear()

            

            # 移除接触点actor
            if hasattr(self, 'contact_points_actor') and self.contact_points_actor:
                try:
                    self.plotter.remove_actor(self.contact_points_actor)
                    self.contact_points_actor = None
                    print("[CRITICAL DEBUG] 移除接触点actor")
                except Exception as contact_e:
                    print(f"[CRITICAL WARNING] 移除接触点actor失败: {contact_e}")
            
            # 注意：特征点由MainWindow类管理，ModelViewer不再直接处理特征点
            
            # 重置数据
            print("[CRITICAL DEBUG] 重置数据...")
            self.reset_data()
            
            # 重新设置可视化（保留坐标轴和网格）
            print("[CRITICAL DEBUG] 重新设置可视化..")
            self._setup_visualization()
            self._add_help_text()
            
            # 安全渲染
            print("[CRITICAL DEBUG] 执行渲染...")
            try:
                self.plotter.render()
                print("[CRITICAL DEBUG] 渲染成功")
            except Exception as render_e:
                print(f"[CRITICAL WARNING] 渲染失败: {render_e}")
                # 使用延迟渲染尝试恢复
                QTimer.singleShot(100, lambda: self.plotter.render())
            
            print("[CRITICAL DEBUG] 安全清除完成")
        except Exception as e:
            print(f"[CRITICAL ERROR] 清除所有模型失败: {e}")
            import traceback
            traceback.print_exc()

    

    def save_screenshot(self, file_path):

        """保存当前视图截图

        

        Args:

            file_path: 保存路径

        """

        try:

            self.plotter.screenshot(file_path)

            print(f"成功保存截图到 {file_path}")

            return True

        except Exception as e:

            print(f"保存截图失败: {e}")

            return False

    

    def reset_view(self):
        """重置视图到默认视角（匹配MeshLab）"""

        print("[CRITICAL DEBUG] 重置视图...")

        try:

            # 重置相机

            print("[CRITICAL DEBUG] 重置相机...")

            self.plotter.reset_camera()

            

            print("[CRITICAL DEBUG] 设置为XY视图...")

            self.plotter.view_xy()

            

            print("[CRITICAL DEBUG] 强制渲染...")

            self.plotter.render()

            

            print("[CRITICAL DEBUG] 视图重置完成")

        except Exception as e:

            print(f"[CRITICAL ERROR] 重置视图失败: {e}")

            import traceback

            traceback.print_exc()

    

    def set_view(self, view_type):

        """设置预定义视�?

        

        Args:

            view_type: 视角类型 ('xy', 'xz', 'yz', 'front', 'back', 'left', 'right', 'top', 'bottom', 'iso')

        """

        try:

            view_type = view_type.lower()

            print(f"[CRITICAL DEBUG] 设置视图�? {view_type}")

            

            # 根据视角类型设置相机位置

            if view_type == 'xy':

                self.plotter.view_xy()

            elif view_type == 'xz':

                self.plotter.view_xz()

            elif view_type == 'yz':

                self.plotter.view_yz()

            elif view_type == 'front':

                self.plotter.view_xz(negative=False)

            elif view_type == 'back':

                self.plotter.view_xz(negative=True)

            elif view_type == 'left':

                self.plotter.view_yz(negative=True)

            elif view_type == 'right':

                self.plotter.view_yz(negative=False)

            elif view_type == 'top':

                self.plotter.view_xy(negative=True)

            elif view_type == 'bottom':

                self.plotter.view_xy(negative=False)

            elif view_type == 'iso':

                # 设置等轴测视�?

                self.plotter.camera_position = 'iso'

            else:

                print(f"未知的视角类�? {view_type}")

                return

            

            # 强制渲染

            self.plotter.render()

            print(f"[CRITICAL DEBUG] 视角 {view_type} 设置完成")

        except Exception as e:

            print(f"[CRITICAL ERROR] 设置视角失败: {e}")

    

    def set_background_color(self, color):

        """设置背景颜色

        

        Args:

            color: 颜色�?

        """

        try:

            self.background_color = color

            self.plotter.set_background(color)

            self.plotter.render()

            print(f"背景颜色已设置为: {color}")

        except Exception as e:

            print(f"设置背景颜色失败: {e}")

    

    def toggle_grid(self, show=True):

        """切换网格显示

        

        Args:

            show: 是否显示网格

        """

        try:

            self.show_grid = show

            # 重新设置可视化参数

            self._setup_visualization()

            self.plotter.render()

            print(f"网格显示: {'开启' if show else '关闭'}")

        except Exception as e:

            print(f"切换网格显示失败: {e}")

    

    def zoom_in(self):

        """放大视图"""

        try:

            self.plotter.camera.Zoom(1.2)  # 放大20%

            self.plotter.render()

            print("视图已放大")

        except Exception as e:

            print(f"放大视图失败: {e}")

    

    def zoom_out(self):

        """缩小视图"""

        try:

            self.plotter.camera.Zoom(0.8)  # 缩小20%

            self.plotter.render()

            print("视图已缩小")

        except Exception as e:

            print(f"缩小视图失败: {e}")

    

    def capture_high_res_screenshot(self, file_path, scale=2):

        """捕获高分辨率截图

        

        Args:

            file_path: 保存路径

            scale: 分辨率缩放因�?

        """

        try:

            # 使用高分辨率设置捕获截图

            self.plotter.screenshot(file_path, window_size=[int(s*scale) for s in self.plotter.window_size])

            print(f"成功保存高分辨率截图�? {file_path}")

            return True

        except Exception as e:

            print(f"保存高分辨率截图失败: {e}")

            return False

    

    def get_models(self):

        """获取所有模�?

        

        Returns:

            dict: 模型字典

        """

        return self.models

    

    def get_original_model(self, model_name):

        """获取原始模型（未简化的模型

        

        Args:

            model_name: 模型名称

            

        Returns:

            pyvista.PolyData: 原始模型

        """

        if model_name in self.original_models:

            return self.original_models[model_name]

        elif model_name in self.models:

            # 如果没有保存原始模型，返回当前模�?

            return self.models[model_name]

        else:

            return None

    

    def get_marked_points(self):

        """获取标记�?

        

        Returns:

            list: 标记点列�?

        """

        return self.marked_points

    

    def get_plane_params(self):

        """获取平面参数

        

        Returns:

            tuple: 平面参数 (a, b, c, d)

        """

        return self.plane_params

        

    def _show_mandible_crest_points(self):

        """显示下颌后槽牙槽嵴标记点"""

        if not hasattr(self, 'mandible_crest_points') or not self.mandible_crest_points:

            return

        

        # 创建点云

        points_array = np.array(self.mandible_crest_points)

        points_mesh = pv.PolyData(points_array)

        

        # 设置橙色�?

        self.marked_points_actor = self.plotter.add_mesh(

            points_mesh,

            color='orange',

            point_size=20.0,

            render_points_as_spheres=True,

            opacity=1.0,

            name="mandible_crest_points",

            reset_camera=False

        )

        

        # 添加标签

        for i, point in enumerate(self.mandible_crest_points):

            side = "左侧" if i == 0 else "右侧"

            self.plotter.add_point_labels(

                [point],

                [f"后槽牙槽嵴{side}"],

                font_size=12,

                show_points=False,

                name=f"mandible_crest_label_{i+1}"

            )

        

        # 刷新视图

        self.plotter.render()

    

    def show_mandible_crest_line(self):

        """显示下颌后槽牙槽嵴连线，生成平滑曲线"""

        if not hasattr(self, 'mandible_crest_points') or len(self.mandible_crest_points) < 2:

            return

        

        # 移除已存在的连线

        if hasattr(self, 'mandible_crest_line_actor') and self.mandible_crest_line_actor:

            self.plotter.remove_actor(self.mandible_crest_line_actor)

            self.mandible_crest_line_actor = None

        

        # 直接使用标记的顺序，不进行排�?

        points_array = np.array(self.mandible_crest_points)

        

        # 数据预处理：点云平滑处理

        # 对原始标记点进行简单平滑，去除异常点影�?

        from scipy.signal import savgol_filter

        if len(points_array) > 3:

            # 使用Savitzky-Golay滤波器进行平面

            window_size = min(5, len(points_array))

            points_array[:, 0] = savgol_filter(points_array[:, 0], window_size, 2)

            points_array[:, 1] = savgol_filter(points_array[:, 1], window_size, 2)

            points_array[:, 2] = savgol_filter(points_array[:, 2], window_size, 2)

        

        # 使用B样条曲线拟合，获得平滑曲线

        from scipy.interpolate import splprep, splev, make_interp_spline

        

        # 提取x, y, z坐标

        x = points_array[:, 0]

        y = points_array[:, 1]

        z = points_array[:, 2]

        

        # 进行B样条曲线拟合，优化平滑因�?

        # 根据点的数量动态调整平滑因子，点数量越多，平滑因子适当增大

        s_factor = max(0.1, min(1.0, len(points_array) / 10.0))

        tck, u = splprep([x, y, z], s=s_factor, k=min(3, len(points_array)-1))

        

        # 生成更多的点来创建平滑曲线

        u_new = np.linspace(u.min(), u.max(), 200)  # 生成200个点，使曲线更平面

        x_new, y_new, z_new = splev(u_new, tck)

        

        # 进一步优化：使用make_interp_spline进行精细插值，提高曲线光滑�?

        try:

            # 创建参数化的t�?

            t_original = np.linspace(0, 1, len(x_new))

            t_fine = np.linspace(0, 1, 400)  # 生成400个精细点

            

            # 使用三次样条进行精细插�?

            spl_x = make_interp_spline(t_original, x_new, k=3)

            spl_y = make_interp_spline(t_original, y_new, k=3)

            spl_z = make_interp_spline(t_original, z_new, k=3)

            

            x_fine = spl_x(t_fine)

            y_fine = spl_y(t_fine)

            z_fine = spl_z(t_fine)

            

            # 组合成最终的平滑点数量

            smooth_points = np.column_stack((x_fine, y_fine, z_fine))

        except Exception as e:

            print(f"[MODEL_VIEWER] 精细插值失败，使用原始平滑�? {e}")

            smooth_points = np.column_stack((x_new, y_new, z_new))

        

        # 创建连线的点和线�?

        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引

        line_segments = []

        for i in range(len(smooth_points) - 1):

            line_segments.extend([2, i, i + 1])

        lines = np.array(line_segments)

        

        # 创建线网�?

        line_mesh = pv.PolyData(smooth_points, lines=lines)

        

        # 保存曲线数据到属性中，供投影生成使用
        self.mandible_crest_curve = smooth_points.tolist()
        print(f"[MODEL_VIEWER] 保存下颌牙槽嵴曲线数据，共 {len(self.mandible_crest_curve)} 个点")
        
        # 显示连线，优化渲染参数
        self.mandible_crest_line_actor = self.plotter.add_mesh(
            line_mesh,
            color='#FF9800',  # 更鲜艳的橙色
            line_width=8.0,  # 适当增加线宽，提高可见�?
            opacity=0.95,  # 轻微降低透明度，增加层次�?
            smooth_shading=True,  # 启用平滑着�?
            name="mandible_crest_line",
            reset_camera=False

        )

        

        print("[MODEL_VIEWER] 已显示下颌后槽牙槽嵴连线")
        
        # 刷新视图
        self.plotter.render()
    
    def show_divide_maxilla_line(self):
        """显示划分上颌的连接线，使用平滑曲线连接所有标记点"""
        if not hasattr(self, 'divide_maxilla_points') or len(self.divide_maxilla_points) < 2:
            return
        
        # 移除已存在的连线
        if hasattr(self, 'divide_maxilla_line_actor') and self.divide_maxilla_line_actor:
            self.plotter.remove_actor(self.divide_maxilla_line_actor)
            self.divide_maxilla_line_actor = None
        
        # 直接使用标记的顺序，不进行排序
        points_array = np.array(self.divide_maxilla_points)
        
        # 数据预处理：点云平滑处理
        # 对原始标记点进行简单平滑，去除异常点影响
        from scipy.signal import savgol_filter
        if len(points_array) > 3:
            # 使用Savitzky-Golay滤波器进行平滑
            window_size = min(5, len(points_array))
            points_array[:, 0] = savgol_filter(points_array[:, 0], window_size, 2)
            points_array[:, 1] = savgol_filter(points_array[:, 1], window_size, 2)
            points_array[:, 2] = savgol_filter(points_array[:, 2], window_size, 2)
        
        # 使用B样条曲线拟合，获得平滑曲线
        from scipy.interpolate import splprep, splev, make_interp_spline
        
        # 提取x, y, z坐标
        x = points_array[:, 0]
        y = points_array[:, 1]
        z = points_array[:, 2]
        
        # 进行B样条曲线拟合，优化平滑因子
        # 根据点的数量动态调整平滑因子，点数量越多，平滑因子适当增大
        s_factor = max(0.1, min(1.0, len(points_array) / 10.0))
        tck, u = splprep([x, y, z], s=s_factor, k=min(3, len(points_array)-1))
        
        # 生成更多的点来创建平滑曲线
        u_new = np.linspace(u.min(), u.max(), 200)  # 生成200个点，使曲线更平滑
        x_new, y_new, z_new = splev(u_new, tck)
        
        # 进一步优化：使用make_interp_spline进行精细插值，提高曲线光滑度
        try:
            # 创建参数化的t值
            t_original = np.linspace(0, 1, len(x_new))
            t_fine = np.linspace(0, 1, 400)  # 生成400个精细点
            
            # 使用三次样条进行精细插值
            spl_x = make_interp_spline(t_original, x_new, k=3)
            spl_y = make_interp_spline(t_original, y_new, k=3)
            spl_z = make_interp_spline(t_original, z_new, k=3)
            
            x_fine = spl_x(t_fine)
            y_fine = spl_y(t_fine)
            z_fine = spl_z(t_fine)
            
            # 组合成最终的平滑点数组
            smooth_points = np.column_stack((x_fine, y_fine, z_fine))
        except Exception as e:
            print(f"[MODEL_VIEWER] 精细插值失败，使用原始平滑点: {e}")
            smooth_points = np.column_stack((x_new, y_new, z_new))
        
        # 创建连线的点和线段
        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引
        line_segments = []
        for i in range(len(smooth_points) - 1):
            line_segments.extend([2, i, i + 1])
        lines = np.array(line_segments)
        
        # 创建线网格
        line_mesh = pv.PolyData(smooth_points, lines=lines)
        
        # 保存平滑曲线到实例属性，以便投影时使用
        self.divide_maxilla_curve = smooth_points.tolist()
        
        # 显示连线，设置为青色
        self.divide_maxilla_line_actor = self.plotter.add_mesh(
            line_mesh,
            color='#00BCD4',  # 青色
            line_width=8.0,  # 适当增加线宽，提高可见度
            opacity=0.95,  # 轻微降低透明度，增加层次感
            smooth_shading=True,  # 启用平滑着色
            name="divide_maxilla_line",
            reset_camera=False
        )
        
        # 划分上颌的标记线不再直接添加到marker_lines，改为在投影时根据当前模式处理
        # 这样可以确保划分上颌的标记线只在特定情况下投影
        
        print(f"[MODEL_VIEWER] 已显示划分上颌的平滑连接线，使用{len(self.divide_maxilla_points)}个点，生成了{len(smooth_points)}个平滑点")
        
        # 刷新视图
        self.plotter.render()
        
    def show_divide_mandible_line(self):
        """显示划分下颌的连接线，使用直线连接所有标记点"""
        if not hasattr(self, 'divide_mandible_points') or len(self.divide_mandible_points) < 2:
            return
        
        # 移除已存在的连线
        if hasattr(self, 'divide_mandible_line_actor') and self.divide_mandible_line_actor:
            self.plotter.remove_actor(self.divide_mandible_line_actor)
            self.divide_mandible_line_actor = None
        
        # 直接使用标记的顺序，不进行排序
        points_array = np.array(self.divide_mandible_points)
        
        # 创建连线的点和线段
        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引
        line_segments = []
        for i in range(len(points_array) - 1):
            line_segments.extend([2, i, i + 1])
        lines = np.array(line_segments)
        
        # 创建线网格
        line_mesh = pv.PolyData(points_array, lines=lines)
        
        # 显示连线，设置为蓝色
        self.divide_mandible_line_actor = self.plotter.add_mesh(
            line_mesh,
            color='#2196F3',  # 蓝色
            line_width=8.0,  # 适当增加线宽，提高可见度
            opacity=0.95,  # 轻微降低透明度，增加层次感
            smooth_shading=True,  # 启用平滑着色
            name="divide_mandible_line",
            reset_camera=False
        )
        
        # 将标记线添加到marker_lines属性，以便生成投影图像时使用
        if not hasattr(self, 'marker_lines') or self.marker_lines is None:
            self.marker_lines = []
        elif not isinstance(self.marker_lines, list):
            self.marker_lines = []
        
        # 添加当前划分下颌的连接线
        self.marker_lines.append(points_array.tolist())
        
        print("[MODEL_VIEWER] 已显示划分下颌的连接线")
        
        # 刷新视图
        self.plotter.render()

    

    def project_point_to_plane(self, point, plane_params):

        """将点投影到平面

        

        Args:

            point: 3D点坐�?(x, y, z)

            plane_params: 平面参数 (a, b, c, d)

            

        Returns:

            投影后的点坐�?(x, y, z)

        """

        a, b, c, d = plane_params

        x, y, z = point

        

        # 计算点到平面的距�?

        distance = (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

        

        # 投影�?= 原始�?- 距离 * 法向量

        projected_x = x - a * distance

        projected_y = y - b * distance

        projected_z = z - c * distance

        

        return (projected_x, projected_y, projected_z)

    

    def project_mandible_crest_to_plane(self):

        """将下颌后槽牙槽嵴连线投影到𬌗平面，并生成平滑曲线"""

        if not hasattr(self, 'mandible_crest_points') or len(self.mandible_crest_points) < 2:

            print("[MODEL_VIEWER] 错误: 下颌后槽牙槽嵴点不足，无法投影")

            return

        

        if not hasattr(self, 'plane_params') or self.plane_params is None:

            print("[MODEL_VIEWER] 错误: 未拟合𬌗平面，无法投影")

            return

        

        # 移除已存在的投影�?

        if hasattr(self, 'mandible_crest_projection_actor') and self.mandible_crest_projection_actor:

            self.plotter.remove_actor(self.mandible_crest_projection_actor)

            self.mandible_crest_projection_actor = None

        

        # 直接使用标记的顺序，不进行排�?

        points_array = np.array(self.mandible_crest_points)

        

        # 数据预处理：点云平滑处理

        # 对原始标记点进行简单平滑，去除异常点影�?

        from scipy.signal import savgol_filter

        if len(points_array) > 3:

            # 使用Savitzky-Golay滤波器进行平面

            window_size = min(5, len(points_array))

            points_array[:, 0] = savgol_filter(points_array[:, 0], window_size, 2)

            points_array[:, 1] = savgol_filter(points_array[:, 1], window_size, 2)

            points_array[:, 2] = savgol_filter(points_array[:, 2], window_size, 2)

        

        # 使用通用投影方法投影点到平面

        projected_points = []

        for point in points_array:

            projected_point = self.project_point_to_plane(point, self.plane_params)

            projected_points.append(projected_point)

        

        projected_points_array = np.array(projected_points)

        

        # 使用B样条曲线拟合，获得平滑曲线

        from scipy.interpolate import splprep, splev, make_interp_spline

        

        # 提取x, y, z坐标

        x = projected_points_array[:, 0]

        y = projected_points_array[:, 1]

        z = projected_points_array[:, 2]

        

        # 进行B样条曲线拟合，优化平滑因�?

        # 根据点的数量动态调整平滑因子，点数量越多，平滑因子适当增大

        s_factor = max(0.1, min(1.0, len(projected_points_array) / 10.0))

        tck, u = splprep([x, y, z], s=s_factor, k=min(3, len(projected_points_array)-1))

        

        # 生成更多的点来创建平滑曲线

        u_new = np.linspace(u.min(), u.max(), 200)  # 生成200个点，使曲线更平面

        x_new, y_new, z_new = splev(u_new, tck)

        

        # 进一步优化：使用make_interp_spline进行精细插值，提高曲线光滑�?

        try:

            # 创建参数化的t�?

            t_original = np.linspace(0, 1, len(x_new))

            t_fine = np.linspace(0, 1, 400)  # 生成400个精细点

            

            # 使用三次样条进行精细插�?

            spl_x = make_interp_spline(t_original, x_new, k=3)

            spl_y = make_interp_spline(t_original, y_new, k=3)

            spl_z = make_interp_spline(t_original, z_new, k=3)

            

            x_fine = spl_x(t_fine)

            y_fine = spl_y(t_fine)

            z_fine = spl_z(t_fine)

            

            # 组合成最终的平滑点数量

            smooth_points = np.column_stack((x_fine, y_fine, z_fine))

        except Exception as e:

            print(f"[MODEL_VIEWER] 精细插值失败，使用原始平滑�? {e}")

            smooth_points = np.column_stack((x_new, y_new, z_new))

        

        # 创建投影线的点和线段

        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引

        line_segments = []

        for i in range(len(smooth_points) - 1):

            line_segments.extend([2, i, i + 1])

        lines = np.array(line_segments)

        

        # 创建线网�?

        line_mesh = pv.PolyData(smooth_points, lines=lines)

        

        # 显示投影线（使用橙色，优化渲染参数）

        self.mandible_crest_projection_actor = self.plotter.add_mesh(

            line_mesh,

            color='#FF9800',  # 使用橙色，与非投影曲线保持一�?

            line_width=8.0,  # 适当增加线宽，提高可见�?

            opacity=0.95,  # 轻微降低透明度，增加层次�?

            smooth_shading=True,  # 启用平滑着�?

            name="mandible_crest_projection",

            reset_camera=False

        )

        

    
        
        # 将投影后的平滑点保存到projected_marker_lines属性
        if len(smooth_points) > 0:
            # 确保projected_marker_lines是列表类型
            if not hasattr(self, 'projected_marker_lines') or not isinstance(self.projected_marker_lines, list):
                self.projected_marker_lines = []
            
            # 移除旧的下颌投影曲线（如果存在）
            self.projected_marker_lines = [line for line in self.projected_marker_lines 
                                          if line.get('type') != 'mandible_crest_projection']
            
            # 添加新的下颌投影曲线
            self.projected_marker_lines.append({
                'type': 'mandible_crest_projection',
                'points': smooth_points.tolist()
            })
        
        print("[MODEL_VIEWER] 已将下颌后槽牙槽嵴连线投影到𬌗平面")
        
        # 刷新视图
        self.plotter.render()
    
    def project_divide_maxilla_to_plane(self):
        """将划分上颌的连接线投影到𬌗平面"""
        if not hasattr(self, 'divide_maxilla_points') or len(self.divide_maxilla_points) < 2:
            print("[MODEL_VIEWER] 错误: 划分上颌点不足，无法投影")
            return
        
        if not hasattr(self, 'plane_params') or self.plane_params is None:
            print("[MODEL_VIEWER] 错误: 未拟合𬌗平面，无法投影")
            return
        
        # 移除已存在的投影线
        if hasattr(self, 'divide_maxilla_projection_actor') and self.divide_maxilla_projection_actor:
            self.plotter.remove_actor(self.divide_maxilla_projection_actor)
            self.divide_maxilla_projection_actor = None
        
        # 直接使用标记的顺序，不进行排序
        points_array = np.array(self.divide_maxilla_points)
        
        # 投影所有点到𬌗平面
        projected_points = []
        for point in points_array:
            projected_point = self.project_point_to_plane(point, self.plane_params)
            projected_points.append(projected_point)
        
        projected_points_array = np.array(projected_points)
        
        # 创建投影线的点和线段（使用直线连接）
        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引
        line_segments = []
        for i in range(len(projected_points_array) - 1):
            line_segments.extend([2, i, i + 1])
        lines = np.array(line_segments)
        
        # 创建线网格
        line_mesh = pv.PolyData(projected_points_array, lines=lines)
        
        # 显示投影线（使用青色）
        self.divide_maxilla_projection_actor = self.plotter.add_mesh(
            line_mesh,
            color='#00BCD4',  # 青色
            line_width=8.0,  # 适当增加线宽，提高可见度
            opacity=0.95,  # 轻微降低透明度，增加层次感
            smooth_shading=True,  # 启用平滑着色
            name="divide_maxilla_projection",
            reset_camera=False
        )
        
        # 将投影后的点保存到projected_marker_lines属性
        if not hasattr(self, 'projected_marker_lines') or self.projected_marker_lines is None:
            self.projected_marker_lines = []
        elif not isinstance(self.projected_marker_lines, list):
            self.projected_marker_lines = []
        
        # 移除已存在的divide_maxilla_projection类型的投影线
        valid_projected_lines = []
        for line in self.projected_marker_lines:
            if isinstance(line, dict) and 'type' in line and line['type'] != 'divide_maxilla_projection':
                valid_projected_lines.append(line)
        self.projected_marker_lines = valid_projected_lines
        
        if len(projected_points_array) > 0:
            self.projected_marker_lines.append({
                'type': 'divide_maxilla_projection',
                'points': projected_points_array.tolist()
            })
        
        print("[MODEL_VIEWER] 已将划分上颌连线投影到𬌗平面")
        
        # 刷新视图
        self.plotter.render()
    
    def project_divide_mandible_to_plane(self):
        """将划分下颌的连接线投影到𬌗平面"""
        if not hasattr(self, 'divide_mandible_points') or len(self.divide_mandible_points) < 2:
            print("[MODEL_VIEWER] 错误: 划分下颌点不足，无法投影")
            return
        
        if not hasattr(self, 'plane_params') or self.plane_params is None:
            print("[MODEL_VIEWER] 错误: 未拟合𬌗平面，无法投影")
            return
        
        # 移除已存在的投影线
        if hasattr(self, 'divide_mandible_projection_actor') and self.divide_mandible_projection_actor:
            self.plotter.remove_actor(self.divide_mandible_projection_actor)
            self.divide_mandible_projection_actor = None
        
        # 直接使用标记的顺序，不进行排序
        points_array = np.array(self.divide_mandible_points)
        
        # 投影所有点到𬌗平面
        projected_points = []
        for point in points_array:
            projected_point = self.project_point_to_plane(point, self.plane_params)
            projected_points.append(projected_point)
        
        projected_points_array = np.array(projected_points)
        
        # 创建投影线的点和线段（使用直线连接）
        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引
        line_segments = []
        for i in range(len(projected_points_array) - 1):
            line_segments.extend([2, i, i + 1])
        lines = np.array(line_segments)
        
        # 创建线网格
        line_mesh = pv.PolyData(projected_points_array, lines=lines)
        
        # 显示投影线（使用蓝色）
        self.divide_mandible_projection_actor = self.plotter.add_mesh(
            line_mesh,
            color='#2196F3',  # 蓝色
            line_width=8.0,  # 适当增加线宽，提高可见度
            opacity=0.95,  # 轻微降低透明度，增加层次感
            smooth_shading=True,  # 启用平滑着色
            name="divide_mandible_projection",
            reset_camera=False
        )
        
        # 将投影后的点保存到projected_marker_lines属性
        if not hasattr(self, 'projected_marker_lines') or self.projected_marker_lines is None:
            self.projected_marker_lines = []
        elif not isinstance(self.projected_marker_lines, list):
            self.projected_marker_lines = []
        
        # 移除已存在的divide_mandible_projection类型的投影线
        valid_projected_lines = []
        for line in self.projected_marker_lines:
            if isinstance(line, dict) and 'type' in line and line['type'] != 'divide_mandible_projection':
                valid_projected_lines.append(line)
        self.projected_marker_lines = valid_projected_lines
        
        if len(projected_points_array) > 0:
            self.projected_marker_lines.append({
                'type': 'divide_mandible_projection',
                'points': projected_points_array.tolist()
            })
        
        print("[MODEL_VIEWER] 已将划分下颌连线投影到𬌗平面")
        
        # 刷新视图
        self.plotter.render()
        
        # 移除已存在的divide_mandible_projection类型的投影线
        valid_projected_lines = []
        for line in self.projected_marker_lines:
            if isinstance(line, dict) and 'type' in line and line['type'] != 'divide_mandible_projection':
                valid_projected_lines.append(line)
        self.projected_marker_lines = valid_projected_lines
        
        if len(projected_points_array) > 0:
            self.projected_marker_lines.append({
                'type': 'divide_mandible_projection',
                'points': projected_points_array.tolist()
            })
        
        print("[MODEL_VIEWER] 已将划分下颌连线投影到𬌗平面")
        
        # 刷新视图
        self.plotter.render()

    

    def show_alveolar_ridge_line(self):
        """显示上颌牙槽嵴连线"""
        if not hasattr(self, 'alveolar_ridge_points') or len(self.alveolar_ridge_points) < 2:
            print("[MODEL_VIEWER] 错误: 上颌牙槽嵴点不足，无法显示连线")
            return

        

        # 调用现有的显示牙槽嵴曲线方法

        self._display_alveolar_ridge_curve()

    

    def project_alveolar_ridge_to_plane(self):
        """将上颌牙槽嵴连线投影到𬌗平面，并生成完整平滑的曲线"""
        if not hasattr(self, 'alveolar_ridge_points') or len(self.alveolar_ridge_points) < 2:
            print("[MODEL_VIEWER] 错误: 上颌牙槽嵴点不足，无法投影")
            return
        
        if not hasattr(self, 'plane_params') or self.plane_params is None:
            print("[MODEL_VIEWER] 错误: 未拟合𬌗平面，无法投影")
            return

        

        # 移除已存在的投影�?

        if hasattr(self, 'alveolar_ridge_projection_actor') and self.alveolar_ridge_projection_actor:

            self.plotter.remove_actor(self.alveolar_ridge_projection_actor)

            self.alveolar_ridge_projection_actor = None

        

        # 直接使用标记的顺序，不进行排�?

        points_array = np.array(self.alveolar_ridge_points)

        

        # 数据预处理：点云平滑处理

        # 对原始标记点进行简单平滑，去除异常点影�?

        from scipy.signal import savgol_filter

        if len(points_array) > 3:

            # 使用Savitzky-Golay滤波器进行平面

            window_size = min(5, len(points_array))

            points_array[:, 0] = savgol_filter(points_array[:, 0], window_size, 2)

            points_array[:, 1] = savgol_filter(points_array[:, 1], window_size, 2)

            points_array[:, 2] = savgol_filter(points_array[:, 2], window_size, 2)

        

        # 获取平面参数 (Ax + By + Cz + D = 0)

        a, b, c, d = self.plane_params

        

        # 计算投影�?

        projected_points = []

        for point in points_array:

            # 计算点到平面的距�?

            distance = (a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)

            

            # 计算投影�?

            projected_point = point - distance * np.array([a, b, c])

            projected_points.append(projected_point)

        

        projected_points_array = np.array(projected_points)

        

        # 使用B样条曲线拟合，获得平滑曲线

        from scipy.interpolate import splprep, splev, make_interp_spline

        

        # 提取x, y, z坐标

        x = projected_points_array[:, 0]

        y = projected_points_array[:, 1]

        z = projected_points_array[:, 2]

        

        # 进行B样条曲线拟合，优化平滑因�?

        # 根据点的数量动态调整平滑因子，点数量越多，平滑因子适当增大

        s_factor = max(0.1, min(1.0, len(projected_points_array) / 10.0))

        tck, u = splprep([x, y, z], s=s_factor, k=min(3, len(projected_points_array)-1))

        

        # 生成更多的点来创建平滑曲线

        u_new = np.linspace(u.min(), u.max(), 200)  # 生成200个点，使曲线更平面

        x_new, y_new, z_new = splev(u_new, tck)

        

        # 进一步优化：使用make_interp_spline进行精细插值，提高曲线光滑�?

        try:

            # 创建参数化的t�?

            t_original = np.linspace(0, 1, len(x_new))

            t_fine = np.linspace(0, 1, 400)  # 生成400个精细点

            

            # 使用三次样条进行精细插�?

            spl_x = make_interp_spline(t_original, x_new, k=3)

            spl_y = make_interp_spline(t_original, y_new, k=3)

            spl_z = make_interp_spline(t_original, z_new, k=3)

            

            x_fine = spl_x(t_fine)

            y_fine = spl_y(t_fine)

            z_fine = spl_z(t_fine)

            

            # 组合成最终的平滑点数量

            smooth_points = np.column_stack((x_fine, y_fine, z_fine))

        except Exception as e:

            print(f"[MODEL_VIEWER] 精细插值失败，使用原始平滑�? {e}")

            smooth_points = np.column_stack((x_new, y_new, z_new))

        

        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引

        line_segments = []

        for i in range(len(smooth_points) - 1):

            line_segments.extend([2, i, i + 1])

        lines = np.array(line_segments)

        

        # 创建线网�?

        line_mesh = pv.PolyData(smooth_points, lines=lines)

        

        # 显示投影线（使用深蓝色，优化渲染参数量

        self.alveolar_ridge_projection_actor = self.plotter.add_mesh(

            line_mesh,

            color='#1976D2',  # 使用深蓝色，与非投影曲线保持一�?

            line_width=8.0,  # 适当增加线宽，提高可见�?

            opacity=0.95,  # 轻微降低透明度，增加层次�?

            smooth_shading=True,  # 启用平滑着�?

            name="alveolar_ridge_projection",

            reset_camera=False

        )

        

        # 将投影后的平滑点保存到projected_marker_lines属性
        if len(smooth_points) > 0:
            # 确保projected_marker_lines是列表类型
            if not hasattr(self, 'projected_marker_lines') or not isinstance(self.projected_marker_lines, list):
                self.projected_marker_lines = []
            
            # 移除旧的上颌投影曲线（如果存在）
            self.projected_marker_lines = [line for line in self.projected_marker_lines 
                                          if line.get('type') != 'alveolar_ridge_projection']
            
            # 添加新的上颌投影曲线
            self.projected_marker_lines.append({
                'type': 'alveolar_ridge_projection',
                'points': smooth_points.tolist()
            })

        

        print("[MODEL_VIEWER] 已将上颌牙槽嵴连线投影到𬌗平面")

        

        # 刷新视图

        self.plotter.render()

    

    def _show_alveolar_ridge_points(self):

        """显示牙槽嵴标记点"""

        if not hasattr(self, 'alveolar_ridge_points') or not self.alveolar_ridge_points:

            return

        

        # 创建点云

        points_array = np.array(self.alveolar_ridge_points)

        points_mesh = pv.PolyData(points_array)

        

        # 设置紫色点，与按钮颜色一�?

        self.marked_points_actor = self.plotter.add_mesh(

            points_mesh,

            color='#9C27B0',

            point_size=20.0,

            render_points_as_spheres=True,

            opacity=1.0,

            name="alveolar_ridge_points",

            reset_camera=False

        )

        

        # 添加标签

        for i, point in enumerate(self.alveolar_ridge_points):

            self.plotter.add_point_labels(

                [point],

                [f"牙槽嵴点{i+1}"],

                font_size=12,

                show_points=False,

                name=f"alveolar_ridge_label_{i+1}"

            )

        

        # 如果有足够的点，显示拟合曲线

        if len(self.alveolar_ridge_points) >= 2:

            self._display_alveolar_ridge_curve()

        

        # 刷新视图

        self.plotter.render()

    

    def _display_alveolar_ridge_curve(self):

        """显示牙槽嵴拟合曲线，生成平滑曲线"""

        if not hasattr(self, 'alveolar_ridge_points') or len(self.alveolar_ridge_points) < 2:

            return

        

        # 移除已存在的曲线

        if hasattr(self, 'alveolar_ridge_curve_actor') and self.alveolar_ridge_curve_actor:

            self.plotter.remove_actor(self.alveolar_ridge_curve_actor)

            self.alveolar_ridge_curve_actor = None

        

        # 直接使用标记的顺序，不进行排�?

        points_array = np.array(self.alveolar_ridge_points)

        

        # 数据预处理：点云平滑处理

        # 对原始标记点进行简单平滑，去除异常点影�?

        from scipy.signal import savgol_filter

        if len(points_array) > 3:

            # 使用Savitzky-Golay滤波器进行平面

            window_size = min(5, len(points_array))

            points_array[:, 0] = savgol_filter(points_array[:, 0], window_size, 2)

            points_array[:, 1] = savgol_filter(points_array[:, 1], window_size, 2)

            points_array[:, 2] = savgol_filter(points_array[:, 2], window_size, 2)

        

        # 使用B样条曲线拟合，获得平滑曲线

        from scipy.interpolate import splprep, splev, make_interp_spline

        

        # 提取x, y, z坐标

        x = points_array[:, 0]

        y = points_array[:, 1]

        z = points_array[:, 2]

        

        # 进行B样条曲线拟合，优化平滑因�?

        # 根据点的数量动态调整平滑因子，点数量越多，平滑因子适当增大

        s_factor = max(0.1, min(1.0, len(points_array) / 10.0))

        tck, u = splprep([x, y, z], s=s_factor, k=min(3, len(points_array)-1))

        

        # 生成更多的点来创建平滑曲线

        u_new = np.linspace(u.min(), u.max(), 200)  # 生成200个点，使曲线更平面

        x_new, y_new, z_new = splev(u_new, tck)

        

        # 进一步优化：使用make_interp_spline进行精细插值，提高曲线光滑�?

        try:

            # 创建参数化的t�?

            t_original = np.linspace(0, 1, len(x_new))

            t_fine = np.linspace(0, 1, 400)  # 生成400个精细点

            

            # 使用三次样条进行精细插�?

            spl_x = make_interp_spline(t_original, x_new, k=3)

            spl_y = make_interp_spline(t_original, y_new, k=3)

            spl_z = make_interp_spline(t_original, z_new, k=3)

            

            x_fine = spl_x(t_fine)

            y_fine = spl_y(t_fine)

            z_fine = spl_z(t_fine)

            

            # 组合成最终的平滑点数量

            smooth_points = np.column_stack((x_fine, y_fine, z_fine))

        except Exception as e:

            print(f"[MODEL_VIEWER] 精细插值失败，使用原始平滑�? {e}")

            smooth_points = np.column_stack((x_new, y_new, z_new))

        

        # PyVista要求的线格式：每条线以线段长度开头，然后是点索引

        line_segments = []

        for i in range(len(smooth_points) - 1):

            line_segments.extend([2, i, i + 1])

        lines = np.array(line_segments)

        

        # 创建线网�?

        line_mesh = pv.PolyData(smooth_points, lines=lines)

        

        # 保存曲线数据到属性中，供投影生成使用
        self.alveolar_ridge_curve = smooth_points.tolist()
        print(f"[MODEL_VIEWER] 保存牙槽嵴曲线数据，共 {len(self.alveolar_ridge_curve)} 个点")
        
        # 显示曲线，优化渲染参数
        self.alveolar_ridge_curve_actor = self.plotter.add_mesh(
            line_mesh,
            color='#1976D2',  # 深蓝色，提高与模型的对比�?
            line_width=8.0,  # 适当增加线宽，提高可见�?
            opacity=0.95,  # 轻微降低透明度，增加层次�?
            smooth_shading=True,  # 启用平滑着�?
            name="alveolar_ridge_curve",

            reset_camera=False

        )

        

        print(f"[MODEL_VIEWER] 已显示牙槽嵴拟合曲线，使用{len(self.alveolar_ridge_points)}个点，生成了{len(smooth_points)}个平滑点")

    

    def calculate_incisive_papilla_anterior(self):

        """计算切牙乳突前-10mm位置，并创建与𬌗平面垂直的面"""

        # 检查是否已标记切牙乳突点和前点

        if self.incisive_papilla_point is None or self.incisive_papilla_anterior_point is None:

            print("[MODEL_VIEWER] 未找到切牙乳突点或前点，无法计算")

            return

        

        # 检查是否已拟合𬌗平面

        if not self.plane_fitted or self.plane_params is None:

            print("[MODEL_VIEWER] 未拟合𬌗平面，无法创建垂直面")

            return

        

        print("[MODEL_VIEWER] 开始计算切牙乳突前位置和创建垂直面...")

        

        # 获取平面参数 (Ax + By + Cz + D = 0)

        a, b, c, d = self.plane_params

        

        # 计算切牙乳突点到前点的向量

        papilla_to_anterior = np.array(self.incisive_papilla_anterior_point) - np.array(self.incisive_papilla_point)

        

        # 计算向量长度，检查是否在8-10mm范围�?

        vector_length = np.linalg.norm(papilla_to_anterior)

        print(f"[MODEL_VIEWER] 切牙乳突点到前点的距�? {vector_length:.2f}mm")

        

        # 归一化向量

        if vector_length > 0:

            papilla_to_anterior_normalized = papilla_to_anterior / vector_length

        else:

            print("[MODEL_VIEWER] 错误：切牙乳突点和前点重叠")

            return

        

        # 创建与𬌗平面垂直且包含前点和平面法向量的平面

        # 垂直面的法向量是𬌗平面法向量和乳突到前点向量的叉积

        plane_normal = np.array([a, b, c])

        cutting_plane_normal = np.cross(plane_normal, papilla_to_anterior_normalized)

        

        # 检查叉积是否为零向量

        if np.linalg.norm(cutting_plane_normal) < 1e-6:

            # 如果叉积为零，说明两个向量平行，使用另一个方向

            print("[MODEL_VIEWER] 警告：叉积为零，调整垂直面方向")

            # 使用与𬌗平面法向量和前点向量都垂直的方向

            if abs(plane_normal[0]) < abs(plane_normal[1]) and abs(plane_normal[0]) < abs(plane_normal[2]):

                # x分量最小，使用x轴方向

                temp_vector = np.array([1, 0, 0])

            elif abs(plane_normal[1]) < abs(plane_normal[2]):

                # y分量最小，使用y轴方向

                temp_vector = np.array([0, 1, 0])

            else:

                # z分量最小，使用z轴方向

                temp_vector = np.array([0, 0, 1])

            cutting_plane_normal = np.cross(plane_normal, temp_vector)

        

        # 归一化切割平面法向量

        cutting_plane_normal = cutting_plane_normal / np.linalg.norm(cutting_plane_normal)

        

        # 计算切割平面方程：ax + by + cz + d = 0

        cutting_a, cutting_b, cutting_c = cutting_plane_normal

        # 使用前点计算d�?

        cutting_d = -np.dot(cutting_plane_normal, np.array(self.incisive_papilla_anterior_point))

        

        # 保存切割平面参数

        self.cutting_plane_params = (cutting_a, cutting_b, cutting_c, cutting_d)

        

        print(f"[MODEL_VIEWER] 切割平面方程：{cutting_a:.6f}x + {cutting_b:.6f}y + {cutting_c:.6f}z + {cutting_d:.6f} = 0")

        

        # 显示对称线（从前点到其在�平面上的投影）

        # 计算前点在𬌗平面上的投影

        def project_point_to_plane(point, plane_params):

            a, b, c, d = plane_params

            point_array = np.array(point)

            # 计算点到平面的距�?

            distance = (a * point_array[0] + b * point_array[1] + c * point_array[2] + d) / np.sqrt(a**2 + b**2 + c**2)

            # 计算投影�?

            projection = point_array - distance * np.array([a, b, c])

            return projection

        

        # 投影前点到�平面

        projection_point = project_point_to_plane(self.incisive_papilla_anterior_point, self.plane_params)

        print(f"[MODEL_VIEWER] 前点在𬌗平面上的投影: {projection_point}")

        

        # 创建对称�?

        line_points = np.array([self.incisive_papilla_anterior_point, projection_point])

        line_mesh = pv.PolyData(line_points)

        

        # 创建线的连接

        lines = np.array([2, 0, 1])  # [线段长度, �?索引, �?索引]

        line_mesh.lines = lines

        

        # 移除已存在的对称�?

        if self.symmetry_line_actor:

            self.plotter.remove_actor(self.symmetry_line_actor)

        

        # 显示对称�?

        self.symmetry_line_actor = self.plotter.add_mesh(

            line_mesh,

            color='purple',

            line_width=4.0,

            opacity=1.0,

            name="symmetry_line",

            reset_camera=False

        )

        

        # 显示切割平面

        # 计算平面的范�?

        # 获取场景中模型的边界范围

        all_points = []

        for model in self.models.values():

            if isinstance(model, pv.PolyData):

                all_points.extend(model.points)

            elif hasattr(model, 'vertices'):

                all_points.extend(np.asarray(model.vertices))

        

        if all_points:

            all_points = np.array(all_points)

            min_coords = np.min(all_points, axis=0)

            max_coords = np.max(all_points, axis=0)

        else:

            # 如果没有模型，使用默认范�?

            min_coords = np.array([-50, -50, -50])

            max_coords = np.array([50, 50, 50])

        

        # 扩展范围

        padding = 20.0

        x_range = (min_coords[0] - padding, max_coords[0] + padding)

        y_range = (min_coords[1] - padding, max_coords[1] + padding)

        

        # 创建网格�?

        xx, yy = np.meshgrid(

            np.linspace(x_range[0], x_range[1], 50),

            np.linspace(y_range[0], y_range[1], 50)

        )

        

        # 计算z坐标

        if abs(cutting_c) > 1e-6:

            zz = (-cutting_a * xx - cutting_b * yy - cutting_d) / cutting_c

        else:

            # 特殊情况处理

            zz = np.zeros_like(xx)

        

        # 创建平面网格

        plane_mesh = pv.StructuredGrid(xx, yy, zz)

        

        # 移除已存在的切割平面

        if self.cutting_plane_actor:

            self.plotter.remove_actor(self.cutting_plane_actor)

        

        # 显示切割平面，使用半透明紫色

        self.cutting_plane_actor = self.plotter.add_mesh(

            plane_mesh,

            color='purple',

            opacity=0.3,

            name="cutting_plane",

            reset_camera=False

        )

        

        # 使用切割平面切割模型并显示切割线

        self.cut_model_with_plane(cutting_plane_normal, self.incisive_papilla_anterior_point)

        

        # 刷新视图

        self.plotter.render()

        print("[MODEL_VIEWER] 对称线和切割平面已显示")

        

    def generate_maxilla_vertical_plane(self):

        """生成垂直于平面的maxilla垂直平面

        

        该平面将�?

        1. 垂直于红色点标记的平面

        2. 通过蓝色点标记的两个牙颌点形成的连线

        """

        # 检查是否已拟合平面

        if not hasattr(self, 'plane_fitted') or not self.plane_fitted or not hasattr(self, 'plane_params'):

            print("[MODEL_VIEWER] 未拟合平面，无法生成垂直平面")

            return

        

        # 检查是否有maxilla标记�?

        if not hasattr(self, 'marked_points') or not hasattr(self, 'marked_points_modes'):

            print("[MODEL_VIEWER] 缺少标记点或标记点模式记录")

            return

        

        # 获取所有maxilla模式的标记点

        maxilla_points = [p for i, p in enumerate(self.marked_points) if self.marked_points_modes[i] == 'maxilla']

        if len(maxilla_points) < 2:

            print("[MODEL_VIEWER] 需要至�?个maxilla标记点来形成连线")

            return

        

        # 使用最后两个maxilla标记点形成连�?

        point1 = np.array(maxilla_points[-2])

        point2 = np.array(maxilla_points[-1])

        

        print(f"[MODEL_VIEWER] 使用两个标记点形成连�? {point1}, {point2}")

        

        # 将紫色线添加到marker_lines中，供投影使�?

        line_points = np.array([point1, point2])

        self.marker_lines = [line_points]  # 重置并添加新的紫色线

        

        # 获取平面参数

        plane_normal = np.array(self.plane_params[:3])

        

        # 计算两个标记点形成的向量

        line_vector = point2 - point1

        

        # 确保向量不为�?

        if np.linalg.norm(line_vector) < 1e-6:

            print("[MODEL_VIEWER] 两个标记点重合，无法形成有效连线")

            return

        

        # 计算垂直平面的法向量，它必须垂直于原始平面的法向量和连线向量

        # 首先，垂直平面必须垂直于原始平面，所以它的法向量必须位于原始平面�?

        # 我们需要一个既垂直于原始平面法向量，又能定义垂直平面的方向

        # 使用原始平面法向量和连线向量的叉积来得到垂直平面的法向量

        vertical_plane_normal = np.cross(plane_normal, line_vector)

        

        # 归一化法向量

        norm = np.linalg.norm(vertical_plane_normal)

        if norm < 1e-6:

            print("[MODEL_VIEWER] 无法计算垂直平面法向量，使用默认方向")

            # 使用另一个方向

            if abs(plane_normal[0]) < abs(plane_normal[1]):

                temp_vector = np.array([0, 0, 1])

            else:

                temp_vector = np.array([0, 1, 0])

            vertical_plane_normal = np.cross(plane_normal, temp_vector)

            norm = np.linalg.norm(vertical_plane_normal)

            vertical_plane_normal = vertical_plane_normal / norm

        else:

            vertical_plane_normal = vertical_plane_normal / norm

        

        # 计算垂直平面的方程：ax + by + cz + d = 0

        # 平面通过两个标记点形成的连线，所以可以使用其中一个点来计算d

        a, b, c = vertical_plane_normal

        d = -np.dot(vertical_plane_normal, point1)

        

        # 保存垂直平面参数

        self.maxilla_vertical_plane_params = (a, b, c, d)

        

        print(f"[MODEL_VIEWER] 垂直平面方程：{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

        

        # 显示垂直平面

        # 计算平面的范围，优化为只覆盖模型的必要部�?

        target_points = []

        

        # 优先使用上颌模型的点

        if 'maxilla' in self.models:

            model = self.models['maxilla']

            if isinstance(model, pv.PolyData):

                target_points = model.points

            elif hasattr(model, 'vertices'):

                target_points = np.asarray(model.vertices)

        # 如果没有上颌模型，使用所有模型的�?

        else:

            all_points = []

            for model in self.models.values():

                if isinstance(model, pv.PolyData):

                    all_points.extend(model.points)

                elif hasattr(model, 'vertices'):

                    all_points.extend(np.asarray(model.vertices))

            if all_points:

                target_points = np.array(all_points)

        

        # 获取模型的边界范�?

        all_points = []

        for model_name, model in self.models.items():

            if isinstance(model, pv.PolyData):

                all_points.extend(model.points)

            elif hasattr(model, 'vertices'):

                all_points.extend(np.asarray(model.vertices))

        

        if all_points:

            all_points = np.array(all_points)

            min_coords = np.min(all_points, axis=0)

            max_coords = np.max(all_points, axis=0)

        else:

            # 如果没有模型，使用默认范�?

            min_coords = np.array([-10, -10, -10])

            max_coords = np.array([10, 10, 10])

        

        # 扩展平面范围 - 使其刚好超过模型

        padding = 1.0  # 最小幅度扩展，确保刚好超过模型

        

        # 计算当前范围

        x_min = min_coords[0] - padding

        x_max = max_coords[0] + padding

        y_min = min_coords[1] - padding

        y_max = max_coords[1] + padding

        

        # 确定宽度（x方向）和长度（y方向量

        original_width = x_max - x_min

        

        # 将宽度修改为原来的八分之一

        width = original_width / 8

        

        # 计算中心位置

        x_center = (x_min + x_max) / 2

        y_center = (y_min + y_max) / 2

        

        # 保持宽度不变，将长度变为八�?

        x_range = (x_center - width/2, x_center + width/2)  # 宽度保持不变

        length = width * 8  # 长度变为八�?

        y_range = (y_center - length/2, y_center + length/2)  # 长度为宽度的8�?

        

        print(f"[MODEL_VIEWER] 平面调整: 宽度={width:.2f}, 长度={length:.2f}（长度为宽度�?倍）")

        

        # 创建网格�?

        xx, yy = np.meshgrid(

            np.linspace(x_range[0], x_range[1], 50),

            np.linspace(y_range[0], y_range[1], 50)

        )

        

        # 计算z坐标

        a, b, c, d = self.maxilla_vertical_plane_params

        if abs(c) > 1e-6:

            zz = (-a * xx - b * yy - d) / c

        else:

            # 特殊情况处理

            zz = np.zeros_like(xx)

        

        # 创建平面网格

        plane_mesh = pv.StructuredGrid(xx, yy, zz)

        

        # 移除已存在的垂直平面

        if hasattr(self, 'maxilla_vertical_plane_actor') and self.maxilla_vertical_plane_actor:

            self.plotter.remove_actor(self.maxilla_vertical_plane_actor)

        

        # 显示垂直平面，使用半透明蓝色

        self.maxilla_vertical_plane_actor = self.plotter.add_mesh(

            plane_mesh,

            color='blue',

            opacity=0.3,

            name="maxilla_vertical_plane",

            reset_camera=False

        )

        

        # 不再生成红色垂直短线
        # self.generate_red_perpendicular_line(point1, point2, plane_normal)

        

        # 刷新视图

        self.plotter.render()

        print("[MODEL_VIEWER] 垂直平面已显示")

        

    def generate_red_perpendicular_line(self, point1, point2, plane_normal):

        """在第一个蓝色标记点前生成红色垂�?

        

        Args:

            point1: 第一个蓝色标记点

            point2: 第二个蓝色标记点

            plane_normal: 原始平面的法向量

        """

        # 计算蓝色线的方向向量

        line_vector = point2 - point1

        line_vector_normalized = line_vector / np.linalg.norm(line_vector)

        

        # 计算在point1前面8mm处的目标�?

        # 固定距离�?mm

        distance = 8.0  # 8mm

        target_point = point1 - line_vector_normalized * distance

        

        # 计算垂直于蓝色线和平面的方向向量

        # 这个方向向量应该垂直于蓝色线方向和平面法向量

        perpendicular_vector = np.cross(line_vector_normalized, plane_normal)

        perpendicular_vector_normalized = perpendicular_vector / np.linalg.norm(perpendicular_vector)

        

        # 确定红色线的长度（例�?0mm�?

        line_length = 10.0

        half_length = line_length / 2

        

        # 计算红色线的两个端点

        line_start = target_point - perpendicular_vector_normalized * half_length

        line_end = target_point + perpendicular_vector_normalized * half_length

        

        # 存储红色垂线的端点，以便在投影时使用

        self.red_perpendicular_line_points = [line_start, line_end]

        

        # 创建红色�?

        line_points = np.array([line_start, line_end])

        line_mesh = pv.PolyData(line_points)

        line_mesh.lines = np.array([2, 0, 1])

        

        # 移除已存在的红色�?

        if hasattr(self, 'red_perpendicular_line_actor') and self.red_perpendicular_line_actor:

            self.plotter.remove_actor(self.red_perpendicular_line_actor)

        

        # 显示红色线，使用不透明红色，线�?.0（适量加粗�?

        self.red_perpendicular_line_actor = self.plotter.add_mesh(

            line_mesh,

            color='red',

            opacity=1.0,

            line_width=5.0,

            name="red_perpendicular_line",

            reset_camera=False

        )

        

        print(f"[MODEL_VIEWER] 已生成红色垂�? 起点={line_start}, 终点={line_end}")

        

    def cut_model_with_plane(self, plane_normal, plane_point):

        """使用平面切割模型并显示切割线

        

        Args:

            plane_normal: 切割平面的法向量

            plane_point: 切割平面上的一点

        """

        print("[MODEL_VIEWER] 开始切割模�?..")

        

        # 确保属性存�?

        if not hasattr(self, 'cutting_line_actor'):

            self.cutting_line_actor = None

        

        # 移除已存在的切割�?

        if self.cutting_line_actor:

            self.plotter.remove_actor(self.cutting_line_actor)

            self.cutting_line_actor = None

        

        # 检查是否有模型

        if not self.models:

            print("[MODEL_VIEWER] 没有模型可切割")

            return

        

        # 尝试切割每个模型

        all_intersection_points = []

        

        for model_name, model in self.models.items():

            if not isinstance(model, pv.PolyData):

                continue

            

            try:

                # 使用PyVista的平面切割功�?

                # 创建切割平面

                cut_plane = pv.Plane(center=plane_point, direction=plane_normal, i_size=200, j_size=200)

                

                # 执行切割

                try:

                    # 尝试使用extract_cells_below方法，这是PyVista的一个功�?

                    slice_result = model.slice(normal=plane_normal, origin=plane_point)

                except Exception as e:

                    print(f"[MODEL_VIEWER] 使用slice方法时出错 {e}，尝试其他方向")

                    # 如果slice方法失败，尝试使用clip方法

                    try:

                        clipped = model.clip(normal=plane_normal, origin=plane_point)

                        # 计算切割边界

                        edges = clipped.extract_feature_edges(feature_edges=False, boundary_edges=True, manifold_edges=False)

                        if edges.n_points > 0:

                            slice_result = edges

                        else:

                            print(f"[MODEL_VIEWER] 无法找到模型 {model_name} 的切割边界")

                            continue

                    except Exception as e2:

                        print(f"[MODEL_VIEWER] 使用clip方法切割模型 {model_name} 时出�? {e2}")

                        continue

                

                # 检查切割结�?

                if slice_result.n_points > 0:

                    # 收集所有交�?

                    all_intersection_points.extend(slice_result.points)

                    print(f"[MODEL_VIEWER] 模型 {model_name} 切割成功，找到{slice_result.n_points} 个交点")

                else:
                    print(f"[MODEL_VIEWER] 模型 {model_name} 与切割平面没有交点")

            except Exception as e:

                print(f"[MODEL_VIEWER] 切割模型 {model_name} 时发生错�? {e}")

                continue

        

        # 如果没有找到交点，尝试备用方向

        if not all_intersection_points:

            print("[MODEL_VIEWER] 尝试备用切割方法...")

            

            for model_name, model in self.models.items():

                if not isinstance(model, pv.PolyData):

                    continue

                

                try:

                    # 计算每个三角形与平面的交�?

                    intersection_points = []

                    

                    # 遍历模型的所有三角形

                    for i in range(0, len(model.faces), 4):  # PyVista的faces格式�?[3, p1, p2, p3, 3, p4, p5, p6, ...]

                        if model.faces[i] != 3:

                            continue  # 只处理三角形�?

                        

                        # 获取三角形的三个顶点

                        v1 = model.points[model.faces[i+1]]

                        v2 = model.points[model.faces[i+2]]

                        v3 = model.points[model.faces[i+3]]

                        

                        # 计算每个边与平面的交�?

                        def line_plane_intersection(p1, p2, plane_normal, plane_point):

                            # 直线方向向量

                            line_dir = p2 - p1

                            # 计算直线与平面的夹角

                            denom = np.dot(plane_normal, line_dir)

                            

                            # 如果直线与平面平行，则无交点

                            if abs(denom) < 1e-6:

                                return None

                            

                            # 计算从p1到平面的向量

                            p1_to_plane = plane_point - p1

                            # 计算交点参数t

                            t = np.dot(plane_normal, p1_to_plane) / denom

                            

                            # 检查交点是否在线段�?

                            if t < 0 or t > 1:

                                return None

                            

                            # 计算交点

                            intersection = p1 + t * line_dir

                            return intersection

                        

                        # 检查三角形的三条边

                        edges = [(v1, v2), (v2, v3), (v3, v1)]

                        edge_intersections = []

                        

                        for edge_start, edge_end in edges:

                            intersection = line_plane_intersection(edge_start, edge_end, plane_normal, plane_point)

                            if intersection is not None:

                                edge_intersections.append(intersection)

                        

                        # 如果有两个交点，添加到交点列�?

                        if len(edge_intersections) >= 2:

                            intersection_points.extend(edge_intersections)

                    

                    # 如果找到交点，添加到总列�?

                    if intersection_points:

                        all_intersection_points.extend(intersection_points)

                        print(f"[MODEL_VIEWER] 备用方法成功，在模型 {model_name} 上找到{len(intersection_points)} 个交点")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 使用备用方法切割模型 {model_name} 时出�? {e}")

                    continue

        

        # 如果找到了交点，显示切割�?

        if all_intersection_points:

            # 将交点转换为NumPy数组

            points_array = np.array(all_intersection_points)

            

            # 创建点云

            points_mesh = pv.PolyData(points_array)

            

            # 尝试对点进行排序以创建连续的�?

            try:

                # 计算点的质心

                centroid = np.mean(points_array, axis=0)

                

                # 计算每个点相对于质心的角�?

                angles = []

                for point in points_array:

                    # 计算点相对于质心的向量

                    vec = point - centroid

                    # 忽略z分量，在xy平面上计算角�?

                    angle = np.arctan2(vec[1], vec[0])

                    angles.append(angle)

                

                # 按角度排�?

                sorted_indices = np.argsort(angles)

                sorted_points = points_array[sorted_indices]

                

                # 创建线的连接

                line_segments = []

                for i in range(len(sorted_points) - 1):

                    line_segments.extend([2, i, i + 1])

                # 连接最后一个点和第一个点以形成闭合曲线

                line_segments.extend([2, len(sorted_points) - 1, 0])

                

                # 创建线网�?

                lines = np.array(line_segments)

                cutting_line_mesh = pv.PolyData(sorted_points, lines=lines)

                

                # 显示切割�?

                self.cutting_line_actor = self.plotter.add_mesh(

                    cutting_line_mesh,

                    color='yellow',

                    line_width=3.0,

                    opacity=1.0,

                    name="cutting_line",

                    reset_camera=False

                )

                

                print(f"[MODEL_VIEWER] 成功显示切割线，使用 {len(sorted_points)} 个点")

            except Exception as e:

                # 如果排序失败，直接显示点

                print(f"[MODEL_VIEWER] 对点进行排序时出错 {e}，直接显示交点")

                self.cutting_line_actor = self.plotter.add_mesh(

                    points_mesh,

                    color='yellow',

                    point_size=8.0,

                    render_points_as_spheres=True,

                    opacity=1.0,

                    name="cutting_points",

                    reset_camera=False

                )

        else:

            print("[MODEL_VIEWER] 未能找到任何切割交点")

        

        # 刷新视图

        self.plotter.render()

        print("[MODEL_VIEWER] 模型切割完成")

        

    def get_cutting_line_points(self):

        """获取切割线上的点

        

        Returns:

            list: 切割线上的点列表

        """

        if hasattr(self, 'cutting_line_actor') and self.cutting_line_actor:

            # 尝试从切割线演员获取�?

            try:

                # 获取切割线网�?

                cutting_line_mesh = self.plotter.actors.get('cutting_line')

                if cutting_line_mesh:

                    return cutting_line_mesh.points.tolist()

            except Exception as e:

                print(f"[MODEL_VIEWER] 获取切割线点时出�? {e}")

        return []

    

    def _uniform_sample_points(self, vertices, target_points=100000, tolerance=0.1):

        """均匀采样点集，增加采样密度和覆盖�?

        

        Args:

            vertices: 原始顶点数组

            target_points: 目标采样点数量

            tolerance: 采样点之间的最小距离容�?

            

        Returns:

            np.ndarray: 均匀采样后的点集

        """

        if len(vertices) <= target_points:

            return vertices

            

        # 计算点云边界面

        min_bounds = np.min(vertices, axis=0)

        max_bounds = np.max(vertices, axis=0)

        

        # 计算采样网格大小

        volume = np.prod(max_bounds - min_bounds)

        grid_size = (volume / target_points) ** (1/3)

        

        # 初始化采样结�?

        sampled_points = []

        

        # 使用网格划分进行均匀采样

        for point in vertices:

            # 计算点在网格中的索引

            grid_idx = np.floor((point - min_bounds) / grid_size).astype(int)

            

            # 检查是否已经在该网格单元中采样过点

            if tuple(grid_idx) not in sampled_points:

                sampled_points.append(tuple(grid_idx))

                

            # 如果达到目标采样数量，提前终�?

            if len(sampled_points) >= target_points:

                break

                

        # 将网格索引转换回实际坐标

        sampled_indices = list(range(len(sampled_points)))

        

        # 如果采样点数量不足，使用随机采样补充

        if len(sampled_points) < target_points:

            remaining_points = target_points - len(sampled_points)

            random_indices = np.random.choice(len(vertices), remaining_points, replace=False)

            sampled_indices += random_indices.tolist()

            

        return vertices[sampled_indices]

        

    def project_maxilla_to_plane(self, maxilla_mesh, plane_params=None, grid_resolution=0.2):

        """将上颌模型投影到指定平面，严格按照以下步骤实现：

        1. 读取三角面片数据

        2. 定义投影平面方程

        3. 计算每个面片顶点到平面的垂直距离

        4. 处理数据用于后续深度图生�?

        

        Args:

            maxilla_mesh: 上颌模型（可以是PyVista网格或Open3D的TriangleMesh对象�?

            plane_params: 可选，自定义投影平面参数(A, B, C, D)，默认使用当前拟合的𬌗平面

            grid_resolution: 可选，网格分辨率（mm），用于自适应降采�?

            

        Returns:

            tuple: (projected_vertices, triangles, depth_values)

                  projected_vertices: 投影后的3D顶点坐标

                  triangles: 三角面片数据

                  depth_values: 每个顶点的深度值（到平面的垂直距离�?

        """

        print("[MODEL_VIEWER] 开始将上颌模型投影到平面..")

        

        # 使用指定的平面参数或默认平面参数

        if plane_params is None:

            if not self.plane_params:

                print("[MODEL_VIEWER] 没有拟合的平面，也没有提供自定义平面参数，无法投影")

                return None

            plane_params = self.plane_params

        

        # 获取平面参数 (Ax + By + Cz + D = 0)

        a, b, c, d = plane_params

        print(f"[MODEL_VIEWER] 平面参数: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")

        

        # Step 1: 读取三角面片数据

        try:

            if isinstance(maxilla_mesh, pv.PolyData):

                # PyVista网格

                vertices = maxilla_mesh.points

                triangles = maxilla_mesh.faces.reshape(-1, 4)[:, 1:4]  # 转换为Nx3的三角面片数量

                print(f"[MODEL_VIEWER] 使用PyVista网格")

            elif hasattr(maxilla_mesh, 'vertices') and hasattr(maxilla_mesh, 'triangles'):

                # Open3D的TriangleMesh对象

                vertices = np.asarray(maxilla_mesh.vertices)

                triangles = np.asarray(maxilla_mesh.triangles)

                print(f"[MODEL_VIEWER] 使用Open3D网格")

            else:

                raise TypeError(f"不支持的模型类型: {type(maxilla_mesh)}")

        except Exception as e:

            print(f"[MODEL_VIEWER] 获取模型数据失败: {e}")

            import traceback

            traceback.print_exc()

            return None

        

        print(f"[MODEL_VIEWER] 读取到{len(vertices)} 个顶点和 {len(triangles)} 个三角面片")

        

        # Step 1.5: 自适应降采样（仅当顶点数量超过阈值时启用�?

        downsample_threshold = 50000  # 降采样阈�?

        if len(vertices) > downsample_threshold:

            print(f"[MODEL_VIEWER] 顶点数量超过 {downsample_threshold}，开始自适应降采样..")

            try:

                # 使用Open3D进行体素降采�?

                pcd = o3d.geometry.PointCloud()

                pcd.points = o3d.utility.Vector3dVector(vertices)

                

                # 体素大小与网格分辨率联动

                voxel_size = grid_resolution / 2.0

                downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

                

                # 更新顶点

                vertices = np.asarray(downsampled_pcd.points)

                print(f"[MODEL_VIEWER] 降采样完成，剩余 {len(vertices)} 个顶点")

            except Exception as e:

                print(f"[MODEL_VIEWER] 降采样失败 {e}")

                import traceback

                traceback.print_exc()

        

        # Step 2: 归一化平面法向量

        normal = np.array([a, b, c], dtype=np.float64)

        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-6:

            print("[MODEL_VIEWER] 平面法向量为零，无法投影")

            return None

        normal = normal / normal_norm  # 单位化法向量

        print(f"[MODEL_VIEWER] 归一化平面法向量: {normal}")

        

        # Step 3: 计算每个顶点到平面的垂直距离（深度值）

        # 点到平面距离公式：|Ax + By + Cz + D| / sqrt(A² + B² + C²)

        # 由于已经归一化，距离公式简化为 |Ax + By + Cz + D|

        depth_values = np.abs(np.dot(vertices, normal) + d)

        print(f"[MODEL_VIEWER] 深度值范围 [{np.min(depth_values):.2f}, {np.max(depth_values):.2f}] mm")

        

        # Step 4: 计算每个顶点的投影位�?

        # 投影公式：P_proj = P - (P·n + d) * n

        distances = np.dot(vertices, normal) + d

        projected_vertices = vertices - distances[:, np.newaxis] * normal

        

        print(f"[MODEL_VIEWER] 投影完成，投影点数量: {len(projected_vertices)}")
        
        # 返回投影结果元组
        return (projected_vertices, triangles, depth_values)

        print(f"[MODEL_VIEWER] 投影点坐标范围 X=[{np.min(projected_vertices[:, 0]):.2f}, {np.max(projected_vertices[:, 0]):.2f}], "

              f"Y=[{np.min(projected_vertices[:, 1]):.2f}, {np.max(projected_vertices[:, 1]):.2f}], "

              f"Z=[{np.min(projected_vertices[:, 2]):.2f}, {np.max(projected_vertices[:, 2]):.2f}]")

        

        return projected_vertices, triangles, depth_values

    

    def capture_depth_image_with_open3d(self, maxilla_mesh=None):

        """使用Open3D的VisualizerWithKeyCallback创建交互式可视化窗口

        允许用户手动调整视角，按C'键捕获当前视角的深度�?

        

        Args:

            maxilla_mesh: 要显示的上颌模型

            

        Returns:

            tuple: (color_image, depth_image, camera_params) �?None

        """

        print("[MODEL_VIEWER] 开始执行Open3D深度图捕获")

        vis = None

        

        try:

            # 确保所有必要的导入都可见

            import cv2

            import json

            from scipy.spatial import KDTree

            import matplotlib.pyplot as plt

            

            print("[MODEL_VIEWER] 导入完成")

            

            # 使用当前的上颌模型或默认模型

            if maxilla_mesh is None:

                if not hasattr(self, 'models') or 'maxilla' not in self.models:

                    print("[MODEL_VIEWER] 错误: 没有找到上颌模型")

                    return None

                maxilla_mesh = self.models['maxilla']['mesh']

                print("[MODEL_VIEWER] 使用默认上颌模型")

            

            # 如果是PyVista模型，转换为Open3D模型

            if isinstance(maxilla_mesh, pv.PolyData):

                print(f"[MODEL_VIEWER] 转换PyVista模型到Open3D，点数 {maxilla_mesh.n_points}")

                o3d_mesh = o3d.geometry.TriangleMesh()

                

                # 安全地设置顶点

                try:

                    points = maxilla_mesh.points

                    if points is None or len(points) == 0:

                        print("[MODEL_VIEWER] 错误: 模型没有有效的点数据")

                        return None

                    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)

                    print("[MODEL_VIEWER] 顶点数据设置成功")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 设置顶点时出错 {e}")

                    return None

                

                # 处理面数量

                try:

                    faces = maxilla_mesh.faces

                    if faces is not None and len(faces) > 0:

                        try:

                            # 重塑faces数组为n×4格式

                            faces_reshaped = faces.reshape(-1, 4)[:, 1:4]

                            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_reshaped)

                            print(f"[MODEL_VIEWER] 成功转换三角面片，数量 {len(faces_reshaped)}")

                        except Exception as e:

                            print(f"[MODEL_VIEWER] 面数据转换错误 {e}")

                            # 即使面数据转换失败，也可以继续（只显示点云）

                except Exception as e:

                    print(f"[MODEL_VIEWER] 访问面数据时出错: {e}")

                

                # 计算法向量

                try:

                    o3d_mesh.compute_vertex_normals()

                    print("[MODEL_VIEWER] 成功计算顶点法线")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 计算法线错误: {e}")

            

            elif isinstance(maxilla_mesh, o3d.geometry.TriangleMesh):

                print("[MODEL_VIEWER] 已使用Open3D格式的模型")

                o3d_mesh = maxilla_mesh

                # 确保有法线

                if not o3d_mesh.has_vertex_normals():

                    try:

                        o3d_mesh.compute_vertex_normals()

                        print("[MODEL_VIEWER] 补充计算顶点法线")

                    except Exception as e:

                        print(f"[MODEL_VIEWER] 计算法线错误: {e}")

            else:

                print(f"[MODEL_VIEWER] 错误: 不支持的模型类型: {type(maxilla_mesh)}")

                return None

            

            # 验证模型数据有效性

            if len(o3d_mesh.vertices) == 0:

                print("[MODEL_VIEWER] 错误: 模型没有顶点数据")

                return None

            

            print(f"[MODEL_VIEWER] 模型验证通过，顶点数量 {len(o3d_mesh.vertices)}")

            

            # 用于存储捕获的图像

            captured_data = {

                'color_image': None,

                'depth_image': None,

                'camera_params': None

            }

            

            def capture_images(vis):

                """按键回调函数，捕获当前视角的图像"""

                print("[MODEL_VIEWER] 开始捕获图像..")

                

                try:

                    # 获取相机参数

                    ctr = vis.get_view_control()

                    if ctr is None:

                        print("[MODEL_VIEWER] 错误: 无法获取视图控制器")

                        return False

                    

                    parameters = ctr.convert_to_pinhole_camera_parameters()

                    intrinsic = parameters.intrinsic

                    extrinsic = parameters.extrinsic

                    

                    # 获取当前视角的颜色图和深度图

                    print("[MODEL_VIEWER] 捕获颜色图像...")

                    color_image = vis.capture_screen_float_buffer(do_render=True)

                    print("[MODEL_VIEWER] 捕获深度图像...")

                    depth_image = vis.capture_depth_float_buffer(do_render=True)

                    

                    # 检查图像是否为空

                    if color_image is None:

                        print("[MODEL_VIEWER] 错误: 捕获的颜色图像为空")

                        return False

                    if depth_image is None:

                        print("[MODEL_VIEWER] 错误: 捕获的深度图像为空")

                        return False

                    

                    print("[MODEL_VIEWER] 图像捕获成功，开始数据转换..")

                    

                    # 转换为NumPy 数组

                    try:

                        color_np = np.asarray(color_image) * 255

                        depth_np = np.asarray(depth_image)

                        color_np = cv2.cvtColor(color_np.astype(np.uint8), cv2.COLOR_BGR2RGB)

                    except Exception as e:

                        print(f"[MODEL_VIEWER] 图像转换错误: {e}")

                        import traceback

                        traceback.print_exc()

                        return False

                    

                    # 保存捕获的数量

                    captured_data['color_image'] = color_np

                    captured_data['depth_image'] = depth_np

                    captured_data['camera_params'] = {

                        'intrinsic': intrinsic,

                        'extrinsic': extrinsic

                    }

                    

                    # 保存字典为JSON文件（可选）

                    try:

                        output_file = 'target_landmarks.json'

                        # 创建一个简单的示例数据

                        sample_data = {

                            'camera_intrinsic': intrinsic.intrinsic_matrix.tolist(),

                            'camera_extrinsic': extrinsic.tolist(),

                            'image_shape': color_np.shape

                        }

                        with open(output_file, 'w') as f:

                            json.dump(sample_data, f, indent=4, ensure_ascii=False)

                        print(f"相机参数已保存到 {output_file}")

                    except Exception as e:

                        print(f"保存JSON文件时出错 {e}")

                    

                    print(f"[MODEL_VIEWER] 图像数据保存成功，颜色图形状: {color_np.shape}，深度图形状: {depth_np.shape}")

                    

                    # 安全销毁窗口

                    try:

                        vis.destroy_window()

                        print("[MODEL_VIEWER] 可视化窗口已销毁")

                    except Exception as e:

                        print(f"[MODEL_VIEWER] 销毁窗口时出错: {e}")

                    

                    return True

                    

                except Exception as e:

                    print(f"[MODEL_VIEWER] 捕获图像时出错：{e}")

                    import traceback

                    traceback.print_exc()

                    return False



            # 创建一个可视化器

            try:

                print("[MODEL_VIEWER] 创建Open3D可视化器...")

                vis = o3d.visualization.VisualizerWithKeyCallback()

                

                # 设置窗口属性

                window_name = "请用鼠标操作旋转3D人脸到正面人脸，并按下小写C键"

                width, height = 800, 1000

                

                print(f"[MODEL_VIEWER] 创建窗口: {window_name}, 大小: {width}x{height}")

                

                # 尝试创建窗口，如果失败则使用备用方法

                try:

                    vis.create_window(window_name=window_name, width=width, height=height)

                    print("[MODEL_VIEWER] 窗口创建成功")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 窗口创建失败: {e}，尝试使用备用参数..")

                    # 尝试不指定窗口名

                    try:

                        vis.create_window(width=width, height=height)

                        print("[MODEL_VIEWER] 备用窗口创建成功")

                    except Exception as e2:

                        print(f"[MODEL_VIEWER] 备用窗口创建也失败 {e2}")

                        return None

                

                # 添加几何模型

                print("[MODEL_VIEWER] 添加几何模型到可视化器..")

                vis.add_geometry(o3d_mesh)

                

                # 设置渲染选项

                render_option = vis.get_render_option()

                if render_option:

                    render_option.light_on = True

                    render_option.mesh_show_back_face = True

                    print("[MODEL_VIEWER] 渲染选项设置完成")

                

                # 设置相机视角

                try:

                    ctr = vis.get_view_control()

                    if ctr:

                        # 设置一个合理的初始视角

                        ctr.set_zoom(0.8)

                        print("[MODEL_VIEWER] 相机视角初始化完成")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 设置相机视角时出错 {e}")

                

                # 注册按键回调 - 同时支持大写和小写C�?

                vis.register_key_callback(ord("C"), capture_images)

                vis.register_key_callback(ord("c"), capture_images)

                print("[MODEL_VIEWER] 按键回调注册完成，请按下'C'键捕获图像")

                

                # 运行可视化器

                try:

                    print("[MODEL_VIEWER] 启动可视化器循环...")

                    vis.run()

                    print("[MODEL_VIEWER] 可视化器循环结束")

                except KeyboardInterrupt:

                    print("[MODEL_VIEWER] 可视化器被用户中断")

                except Exception as e:

                    print(f"[MODEL_VIEWER] 可视化器运行出错: {e}")

                    import traceback

                    traceback.print_exc()

                finally:

                    # 确保窗口被销毁

                    if vis is not None:

                        try:

                            vis.destroy_window()

                            print("[MODEL_VIEWER] 可视化器窗口已销毁")

                        except Exception as e:

                            print(f"[MODEL_VIEWER] 最终销毁窗口时出错: {e}")

                

            except Exception as e:

                print(f"[MODEL_VIEWER] 创建或运行可视化器时出错: {e}")

                import traceback

                traceback.print_exc()

                # 确保窗口被销毁

                if vis is not None:

                    try:

                        vis.destroy_window()

                    except:

                        pass

                return None

            

            # 检查是否成功捕获图�?

            if captured_data['color_image'] is not None and captured_data['depth_image'] is not None:

                print("[MODEL_VIEWER] 深度图捕获任务完全完成")

                return captured_data['color_image'], captured_data['depth_image'], captured_data['camera_params']

            else:

                print("[MODEL_VIEWER] 警告: 未能成功捕获图像数据")

                return None

                

        except Exception as e:

            print(f"[MODEL_VIEWER] 深度图捕获失败 {e}")

            import traceback

            traceback.print_exc()

            # 确保窗口被销毁

            if vis is not None:

                try:

                    vis.destroy_window()

                except:

                    pass

            return None

    

    def highlight_point(self, point_3d, color=[1, 0, 0], size=3.0):

        """�?D视图中高亮显示指定点

        

        Args:

            point_3d: 要高亮显示的3D点坐�?

            color: 高亮点的颜色，默认为红色

            size: 高亮点的大小，默认为3.0

        """

        try:

            # 移除已存在的高亮�?

            if hasattr(self, 'highlighted_point_actor') and self.highlighted_point_actor:

                self.plotter.remove_actor(self.highlighted_point_actor)

                self.highlighted_point_actor = None

            

            # 创建高亮点云数据

            point_array = np.array([point_3d])

            points_mesh = pv.PolyData(point_array)

            

            # 添加高亮�?

            self.highlighted_point_actor = self.plotter.add_mesh(

                points_mesh,

                color=color,

                point_size=size,

                render_points_as_spheres=True,

                opacity=1.0,

                name="highlighted_point",

                reset_camera=False

            )

            

            # 渲染更新

            self.plotter.render()

            print(f"[MODEL_VIEWER] 成功高亮显示点 {point_3d}")

            

        except Exception as e:

            print(f"[MODEL_VIEWER ERROR] 高亮显示点失败 {e}")

            import traceback

            traceback.print_exc()

    

    def generate_preview_projection(self, marked_points):

        """生成实时预览投影数据

        

        Args:

            marked_points: 标记点列�?

            

        Returns:

            tuple: (points_3d, depth_values) �?None

        """

        print("[MODEL_VIEWER] 生成实时预览投影数据...")

        

        # 检查是否有上颌模型

        if 'maxilla' not in self.models:

            print("[MODEL_VIEWER] 没有加载上颌模型，无法生成预览")

            return None

        

        # 获取上颌模型

        maxilla_mesh = self.models['maxilla']

        

        # 使用简化的投影方法生成预览数据

        try:

            # 检查是否有拟合的平面

            if not self.plane_params:

                # 如果没有拟合的平面，使用默认平面

                # 使用模型的底面作为默认平面

                print("[MODEL_VIEWER] 没有拟合的平面，使用默认平面")

                

                # 计算模型的包围盒

                if isinstance(maxilla_mesh, pv.PolyData):

                    bounds = maxilla_mesh.bounds

                elif hasattr(maxilla_mesh, 'get_axis_aligned_bounding_box'):

                    bbox = maxilla_mesh.get_axis_aligned_bounding_box()

                    bounds = [bbox.min_bound[0], bbox.max_bound[0],

                             bbox.min_bound[1], bbox.max_bound[1],

                             bbox.min_bound[2], bbox.max_bound[2]]

                else:

                    print("[MODEL_VIEWER] 无法获取模型包围盒")

                    return None

                

                # 使用底面作为默认平面 (Z = min_z)

                min_z = bounds[4]

                self.plane_params = (0, 0, 1, -min_z)  # z - min_z = 0

            

            # 使用快速投影方向

            # 获取模型顶点

            if isinstance(maxilla_mesh, pv.PolyData):

                vertices = maxilla_mesh.points

            elif hasattr(maxilla_mesh, 'vertices'):

                vertices = np.asarray(maxilla_mesh.vertices)

            else:

                print("[MODEL_VIEWER] 无法获取模型顶点")

                return None

            

            # 对顶点进行降采样以提高速度

            if len(vertices) > 10000:

                # 随机采样10000个点

                indices = np.random.choice(len(vertices), 10000, replace=False)

                vertices = vertices[indices]

            

            # 计算深度�?

            a, b, c, d = self.plane_params

            normal = np.array([a, b, c])

            normal = normal / np.linalg.norm(normal)  # 归一�?

            

            # 深度值是点到平面的垂直距�?

            depth_values = np.abs(np.dot(vertices, normal) + d)

            

            print(f"[MODEL_VIEWER] 生成预览数据完成，顶点数量 {len(vertices)}")

            return (vertices, depth_values)

            

        except Exception as e:

            print(f"[MODEL_VIEWER] 生成预览数据失败: {e}")

            import traceback

            traceback.print_exc()

            return None

    

    def convert_3d_to_2d(self, points_3d):

        """�?D点转换为2D坐标，使用基于平面法向量的局部坐标系构建

        

        Args:

            points_3d: 3D点坐标数量

            

        Returns:

            np.ndarray: 2D点坐标数量

        """

        print("[MODEL_VIEWER] 开始3D到2D坐标转换...")

        

        # 检查平面参数

        if not self.plane_params:

            print("[MODEL_VIEWER] 没有拟合的平面，无法转换坐标")

            return None

        

        # 检查标记点数量

        if len(self.marked_points) < 3:

            print(f"[MODEL_VIEWER] 标记点数量不足，需要个点，当前有{len(self.marked_points)}个点")

            return None

        

        # 检查3D点数组是否为空

        if points_3d is None or len(points_3d) == 0:

            print("[MODEL_VIEWER] 3D点数组为空，无法转换坐标")

            return None

        

        # 确保points_3d是numpy数组

        if not isinstance(points_3d, np.ndarray):

            points_3d = np.array(points_3d, dtype=np.float64)  # 使用双精度提高精度

        

        # 检查points_3d的维�?

        if len(points_3d.shape) != 2 or points_3d.shape[1] != 3:

            print(f"[MODEL_VIEWER] 3D点数组维度错误，应为(N, 3)，实际为{points_3d.shape}")

            return None

        

        # 获取平面参数

        a, b, c, d = self.plane_params

        

        # 计算平面的法向量并归一�?

        normal = np.array([a, b, c], dtype=np.float64)

        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-6:

            print("[MODEL_VIEWER] 平面法向量为零，无法转换坐标")

            return None

        normal = normal / normal_norm  # 归一�?

        

        # 检查法向量是否为有效向量

        if np.isnan(normal).any() or np.isinf(normal).any():

            print("[MODEL_VIEWER] 平面法向量包含无效值，无法转换坐标")

            return None

        

        # 创建平面的局部坐标系

        marked_points = np.array(self.marked_points, dtype=np.float64)

        

        # 基于标记点构建稳定的局部坐标系

        # 计算标记点的质心作为原点

        origin = np.mean(marked_points, axis=0)

        

        # 计算协方差矩阵以找到主方向

        offsets = marked_points - origin

        cov_matrix = np.cov(offsets.T, dtype=np.float64)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        

        # 选择最大特征值对应的特征向量作为x轴方向

        x_axis = eigenvectors[:, np.argmax(eigenvalues)]

        

        # 确保x轴在平面�?

        x_axis_proj = x_axis - np.dot(x_axis, normal) * normal

        x_axis_proj_norm = np.linalg.norm(x_axis_proj)



        if x_axis_proj_norm < 1e-6:

            # 如果投影后x轴接近零，使用第二个最大特征值的特征向量

            second_max_idx = np.argsort(eigenvalues)[-2]

            x_axis = eigenvectors[:, second_max_idx]

            x_axis_proj = x_axis - np.dot(x_axis, normal) * normal

            x_axis_proj_norm = np.linalg.norm(x_axis_proj)

            if x_axis_proj_norm < 1e-6:

                print("[MODEL_VIEWER] 无法找到合适的x轴方向")

                return None

        x_axis = x_axis_proj / x_axis_proj_norm  # 单位�?

        

        # 计算y轴（在平面内且与x轴垂直）

        y_axis = np.cross(normal, x_axis)

        y_axis_norm = np.linalg.norm(y_axis)

        if y_axis_norm < 1e-6:

            print("[MODEL_VIEWER] 无法计算y轴，可能是法向量与x轴平面")

            return None

        y_axis = y_axis / y_axis_norm  # 单位�?

        

        # 验证坐标系的正交性和单位�?

        orthogonality = np.dot(x_axis, y_axis)

        coordinate_system_quality = {

            'x_axis_norm': np.linalg.norm(x_axis),

            'y_axis_norm': np.linalg.norm(y_axis),

            'orthogonality': abs(orthogonality)

        }

        print(f"[MODEL_VIEWER] 坐标系质量 {coordinate_system_quality}")

        

        # �?D点转换为2D坐标

        points_offsets = points_3d - origin

        x_coords = np.dot(points_offsets, x_axis)

        y_coords = np.dot(points_offsets, y_axis)

        

        points_2d_array = np.column_stack((x_coords, y_coords))

        

        # 添加转换质量检�?

        # 检�?D点的分布范围

        x_range = np.max(points_2d_array[:, 0]) - np.min(points_2d_array[:, 0])

        y_range = np.max(points_2d_array[:, 1]) - np.min(points_2d_array[:, 1])

        

        if x_range < 1e-6 or y_range < 1e-6:

            print("[MODEL_VIEWER] 警告：3D点分布过于集中，可能存在坐标系构建问题")

        

        print(f"[MODEL_VIEWER] 3D到2D坐标转换完成，输入点数量: {len(points_3d)}, 输出点数量 {len(points_2d_array)}")

        print(f"[MODEL_VIEWER] 2D坐标范围: X=[{np.min(points_2d_array[:, 0]):.6f}, {np.max(points_2d_array[:, 0]):.6f}], "

              f"Y=[{np.min(points_2d_array[:, 1]):.6f}, {np.max(points_2d_array[:, 1]):.6f}]")

        

        return points_2d_array

    def project_mandible_to_plane(self, mandible_mesh, plane_params=None, grid_resolution=0.2):
        """将下颌模型投影到指定平面，严格按照以下步骤实现：

        1. 读取三角面片数据

        2. 定义投影平面方程

        3. 计算每个面片顶点到平面的垂直距离

        4. 处理数据用于后续深度图生成
        

        Args:

            mandible_mesh: 下颌模型（可以是PyVista网格或Open3D的TriangleMesh对象）
            plane_params: 可选，自定义投影平面参数(A, B, C, D)，默认使用当前拟合的𬌗平面
            grid_resolution: 可选，网格分辨率（mm），用于自适应降采样
            

        Returns:

            tuple: (projected_vertices, triangles, depth_values)
                  projected_vertices: 投影后的3D顶点坐标
                  triangles: 三角面片数据
                  depth_values: 每个顶点的深度值（到平面的垂直距离）
        """
        print("[MODEL_VIEWER] 开始将下颌模型投影到平面..")

        # 使用指定的平面参数或默认平面参数
        if plane_params is None:
            if not self.plane_params:
                print("[MODEL_VIEWER] 没有拟合的平面，也没有提供自定义平面参数，无法投影")
                return None
            plane_params = self.plane_params

        # 获取平面参数 (Ax + By + Cz + D = 0)
        a, b, c, d = plane_params
        print(f"[MODEL_VIEWER] 平面参数: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")

        # Step 1: 读取三角面片数据
        try:
            if isinstance(mandible_mesh, pv.PolyData):
                # PyVista网格
                vertices = mandible_mesh.points
                triangles = mandible_mesh.faces.reshape(-1, 4)[:, 1:4]  # 转换为Nx3的三角面片数量
                print(f"[MODEL_VIEWER] 使用PyVista网格")
            elif hasattr(mandible_mesh, 'vertices') and hasattr(mandible_mesh, 'triangles'):
                # Open3D的TriangleMesh对象
                vertices = np.asarray(mandible_mesh.vertices)
                triangles = np.asarray(mandible_mesh.triangles)
                print(f"[MODEL_VIEWER] 使用Open3D网格")
            else:
                raise TypeError(f"不支持的模型类型: {type(mandible_mesh)}")
        except Exception as e:
            print(f"[MODEL_VIEWER] 获取模型数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        print(f"[MODEL_VIEWER] 读取到{len(vertices)} 个顶点和 {len(triangles)} 个三角面片")

        # 跳过降采样步骤，避免三角形信息丢失
        print("[MODEL_VIEWER] 跳过降采样步骤，直接使用原始顶点进行投影计算")

        # Step 2: 归一化平面法向量
        normal = np.array([a, b, c], dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            print("[MODEL_VIEWER] 平面法向量为零，无法投影")
            return None
        normal = normal / normal_norm  # 单位化法向量
        print(f"[MODEL_VIEWER] 归一化平面法向量: {normal}")

        # Step 3: 计算每个顶点到平面的垂直距离（深度值）
        # 点到平面距离公式：|Ax + By + Cz + D| / sqrt(A² + B² + C²)
        # 由于已经归一化，距离公式简化为 |Ax + By + Cz + D|
        depth_values = np.abs(np.dot(vertices, normal) + d)
        print(f"[MODEL_VIEWER] 深度值范围 [{np.min(depth_values):.2f}, {np.max(depth_values):.2f}] mm")

        # Step 4: 计算每个顶点的投影位置
        # 投影公式：P_proj = P - (P·n + d) * n
        distances = np.dot(vertices, normal) + d
        projected_vertices = vertices - distances[:, np.newaxis] * normal
        
        print(f"[MODEL_VIEWER] 投影完成，投影点数量: {len(projected_vertices)}")
        print(f"[MODEL_VIEWER] 投影点坐标范围 X=[{np.min(projected_vertices[:, 0]):.2f}, {np.max(projected_vertices[:, 0]):.2f}], "
              f"Y=[{np.min(projected_vertices[:, 1]):.2f}, {np.max(projected_vertices[:, 1]):.2f}], "
              f"Z=[{np.min(projected_vertices[:, 2]):.2f}, {np.max(projected_vertices[:, 2]):.2f}]")

        return projected_vertices, triangles, depth_values

