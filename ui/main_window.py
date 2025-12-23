import sys
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QFileDialog, QGroupBox,
                              QMessageBox, QProgressBar, QComboBox, QCheckBox,
                              QSlider, QDoubleSpinBox, QSplitter, QFrame, QStatusBar,
                              QStackedWidget, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QThreadPool, QDateTime
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtGui import QFont
import open3d as o3d
import pyvista as pv

# 配置日志系统，与主程序保持一致
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MainWindow')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入处理模块
try:
    from processing.model_loader import ModelLoader
    from ui.model_viewer import ModelViewer
    from projection_depth_code import ProjectionDepthGenerator
    print("成功导入处理模块")
except ImportError as e:
    print(f"导入处理模块时出错: {e}")
    # 确保ModelViewer在导入失败时也能被定义
    ModelViewer = None


class DepthAnalyzer:
    """深度图分析工具类"""
    
    def __init__(self, depth_image, extent):
        self.depth_image = depth_image
        self.extent = extent
        self.x_min, self.x_max, self.y_min, self.y_max = extent
        self.resolution = (self.x_max - self.x_min) / (self.depth_image.shape[1] - 1)
    
    def calculate_section_profile(self, x1, y1, x2, y2, num_points=100):
        """计算剖面线的深度分布"""
        # 生成剖面线上的点
        x = np.linspace(x1, x2, num_points)
        y = np.linspace(y1, y2, num_points)
        
        # 计算点在深度图中的像素坐标
        col = ((x - self.x_min) / (self.x_max - self.x_min) * (self.depth_image.shape[1] - 1)).astype(int)
        row = ((y - self.y_min) / (self.y_max - self.y_min) * (self.depth_image.shape[0] - 1)).astype(int)
        
        # 确保坐标在图像范围内
        col = np.clip(col, 0, self.depth_image.shape[1] - 1)
        row = np.clip(row, 0, self.depth_image.shape[0] - 1)
        
        # 获取深度值
        depth_values = self.depth_image[row, col]
        
        # 计算距离
        distances = np.sqrt((x - x1)**2 + (y - y1)**2)
        
        return distances, depth_values
    
    def calculate_region_statistics(self, x_min, y_min, x_max, y_max):
        """计算指定区域的统计信息"""
        # 转换为图像坐标
        col_min = int((x_min - self.x_min) / self.resolution)
        col_max = int((x_max - self.x_min) / self.resolution)
        row_min = int((y_min - self.y_min) / self.resolution)
        row_max = int((y_max - self.y_min) / self.resolution)
        
        # 确保坐标在图像范围内
        col_min = max(0, col_min)
        col_max = min(self.depth_image.shape[1] - 1, col_max)
        row_min = max(0, row_min)
        row_max = min(self.depth_image.shape[0] - 1, row_max)
        
        # 获取区域深度值
        region_depth = self.depth_image[row_min:row_max+1, col_min:col_max+1]
        
        # 计算统计信息
        stats = {
            'min': np.min(region_depth),
            'max': np.max(region_depth),
            'mean': np.mean(region_depth),
            'std': np.std(region_depth),
            'area': (x_max - x_min) * (y_max - y_min),
            'points': region_depth.size
        }
        
        return stats


class ColorMapManager:
    """色彩映射管理类"""
    
    def __init__(self):
        # 扩展可用的颜色映射，特别增加适合深度图显示的选项
        self.available_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                                   'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
                                   'gray', 'jet', 'rainbow', 'bwr', 'seismic',
                                   'bone', 'terrain', 'gist_earth', 'ocean', 'gist_stern',
                                   'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                                   'gist_rainbow', 'nipy_spectral', 'gist_ncar']
        
        # 推荐的深度图颜色映射
        self.recommended_depth_colormaps = ['viridis', 'plasma', 'inferno', 'cividis', 
                                           'bone', 'terrain', 'jet', 'seismic']
        
    def get_colormap(self, name, reversed=False):
        """获取色彩映射"""
        cmap = plt.get_cmap(name)
        if reversed:
            cmap = cmap.reversed()
        return cmap
    
    def create_custom_colormap(self, colors, positions=None):
        """创建自定义色彩映射"""
        if positions is None:
            positions = np.linspace(0, 1, len(colors))
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
        return cmap
        
    def get_recommended_depth_colormaps(self):
        """获取推荐的深度图颜色映射"""
        return self.recommended_depth_colormaps


class DepthPreviewGenerator:
    """实时预览生成器"""
    
    def __init__(self):
        self.cache = {}
    
    def generate_preview(self, points_3d, depth_values, grid_resolution=0.5):
        """生成快速预览深度图"""
        # 生成缓存键
        cache_key = (len(points_3d), grid_resolution)
        
        # 如果缓存中有相同的结果，直接返回
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 转换为2D点
        points_2d = np.array([(p[0], p[1]) for p in points_3d])
        
        # 计算范围
        x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
        y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
        
        # 创建网格
        xi = np.arange(x_min - 2, x_max + 2, grid_resolution)
        yi = np.arange(y_min - 2, y_max + 2, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # 使用快速插值算法
        from scipy.interpolate import griddata
        zi = griddata(points_2d, depth_values, (xi, yi), method='linear')
        
        # 填充NaN值
        if np.any(np.isnan(zi)):
            zi_nn = griddata(points_2d, depth_values, (xi, yi), method='nearest')
            zi[np.isnan(zi)] = zi_nn[np.isnan(zi)]
        
        # 保存到缓存
        self.cache[cache_key] = (zi, [x_min - 2, x_max + 2, y_min - 2, y_max + 2])
        
        return zi, [x_min - 2, x_max + 2, y_min - 2, y_max + 2]
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()


class EnhancedDepthImageDialog(QWidget):
    """增强的深度图像对话框"""
    
    def __init__(self, depth_image, extent, title="投影深度图", marker_lines_2d=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.depth_image = depth_image.copy()
        self.original_depth_image = depth_image.copy()
        self.extent = extent
        self.marker_lines_2d = marker_lines_2d
        self.current_colormap = 'viridis'
        self.colormap_reversed = False
        self.contrast_min = np.min(self.depth_image)
        self.contrast_max = np.max(self.depth_image)
        
        # 初始化分析工具
        self.analyzer = DepthAnalyzer(self.depth_image, self.extent)
        self.cmap_manager = ColorMapManager()
        
        # 设置窗口大小
        self.resize(1000, 800)
        
        # 创建UI
        self.setup_ui()
        self.setup_connections()
        
        # 绘制初始深度图
        self.update_depth_map()
    
    def closeEvent(self, event):
        """处理对话框关闭事件"""
        # 通知父窗口重新显示三维模型
        if self.parent() and hasattr(self.parent(), 'show_all_models'):
            self.parent().show_all_models()
        event.accept()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 视图切换
        self.view_stack = QStackedWidget()
        
        # 单视图
        self.single_view = self.create_single_view()
        self.view_stack.addWidget(self.single_view)
        
        # 多视图
        self.multi_view = self.create_multi_view()
        self.view_stack.addWidget(self.multi_view)
        
        main_layout.addWidget(self.view_stack)
        
        # 分析面板
        self.analysis_panel = self.create_analysis_panel()
        main_layout.addWidget(self.analysis_panel)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        
        # 视图切换按钮
        self.single_view_btn = QPushButton("单视图")
        self.single_view_btn.setCheckable(True)
        self.single_view_btn.setChecked(True)
        
        self.multi_view_btn = QPushButton("多视图")
        self.multi_view_btn.setCheckable(True)
        
        # 对比度控制
        contrast_label = QLabel("对比度:")
        self.contrast_min_slider = QSlider(Qt.Horizontal)
        self.contrast_max_slider = QSlider(Qt.Horizontal)
        
        # 设置滑块范围
        min_val = np.min(self.depth_image)
        max_val = np.max(self.depth_image)
        slider_range = max_val - min_val
        
        self.contrast_min_slider.setRange(int(min_val * 100), int(max_val * 100))
        self.contrast_max_slider.setRange(int(min_val * 100), int(max_val * 100))
        
        self.contrast_min_slider.setValue(int(self.contrast_min * 100))
        self.contrast_max_slider.setValue(int(self.contrast_max * 100))
        
        # 色彩映射选择
        cmap_label = QLabel("色彩映射:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.cmap_manager.available_colormaps)
        self.colormap_combo.setCurrentText(self.current_colormap)
        
        # 色彩映射反转
        self.colormap_reverse_cb = QCheckBox("反转")
        
        # 自动对比度按钮
        self.auto_contrast_btn = QPushButton("自动对比度")
        
        # 添加到工具栏
        toolbar_layout.addWidget(self.single_view_btn)
        toolbar_layout.addWidget(self.multi_view_btn)
        toolbar_layout.addWidget(contrast_label)
        toolbar_layout.addWidget(self.contrast_min_slider)
        toolbar_layout.addWidget(self.contrast_max_slider)
        toolbar_layout.addWidget(cmap_label)
        toolbar_layout.addWidget(self.colormap_combo)
        toolbar_layout.addWidget(self.colormap_reverse_cb)
        toolbar_layout.addWidget(self.auto_contrast_btn)
        
        return toolbar
    
    def create_single_view(self):
        """创建单视图布局"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建Matplotlib画布
        self.figure_single = Figure(figsize=(8, 6), dpi=120)
        self.canvas_single = FigureCanvas(self.figure_single)
        layout.addWidget(self.canvas_single)
        
        # 创建子图
        self.ax_single = self.figure_single.add_subplot(111)
        
        return widget
    
    def create_multi_view(self):
        """创建多视图布局"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # 创建4个视图
        self.views = []
        self.canvases = []
        
        for i in range(2):
            for j in range(2):
                figure = Figure(figsize=(4, 3), dpi=100)
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111)
                
                self.views.append(ax)
                self.canvases.append(canvas)
                
                layout.addWidget(canvas, i, j)
        
        return widget
    
    def create_analysis_panel(self):
        """创建分析面板"""
        panel = QGroupBox("深度分析工具")
        layout = QHBoxLayout(panel)
        
        # 剖面线分析
        section_group = QGroupBox("剖面线分析")
        section_layout = QVBoxLayout(section_group)
        
        self.section_profile_btn = QPushButton("开始剖面线分析")
        self.section_profile_view = QWidget()
        self.section_profile_view.hide()
        
        section_layout.addWidget(self.section_profile_btn)
        section_layout.addWidget(self.section_profile_view)
        
        # 区域统计
        region_group = QGroupBox("区域统计")
        region_layout = QVBoxLayout(region_group)
        
        self.region_stats_btn = QPushButton("开始区域统计")
        self.region_stats_display = QLabel("统计信息将显示在这里")
        
        region_layout.addWidget(self.region_stats_btn)
        region_layout.addWidget(self.region_stats_display)
        
        # 测量工具
        measure_group = QGroupBox("测量工具")
        measure_layout = QVBoxLayout(measure_group)
        
        self.distance_measure_btn = QPushButton("距离测量")
        self.angle_measure_btn = QPushButton("角度测量")
        
        measure_layout.addWidget(self.distance_measure_btn)
        measure_layout.addWidget(self.angle_measure_btn)
        
        # 导出工具
        export_group = QGroupBox("导出功能")
        export_layout = QVBoxLayout(export_group)
        
        self.export_btn = QPushButton("导出深度图")
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG", "SVG", "TIFF", "PDF", "RAW"])
        
        export_layout.addWidget(QLabel("导出格式:"))
        export_layout.addWidget(self.export_format_combo)
        export_layout.addWidget(self.export_btn)
        
        layout.addWidget(section_group)
        layout.addWidget(region_group)
        layout.addWidget(measure_group)
        layout.addWidget(export_group)
        
        return panel
    
    def setup_connections(self):
        """设置信号连接"""
        # 视图切换
        self.single_view_btn.clicked.connect(self.on_single_view_clicked)
        self.multi_view_btn.clicked.connect(self.on_multi_view_clicked)
        
        # 对比度控制
        self.contrast_min_slider.valueChanged.connect(self.adjust_contrast)
        self.contrast_max_slider.valueChanged.connect(self.adjust_contrast)
        
        # 色彩映射
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        self.colormap_reverse_cb.stateChanged.connect(self.change_colormap)
        
        # 自动对比度
        self.auto_contrast_btn.clicked.connect(self.apply_auto_contrast)
        
        # 分析工具
        self.section_profile_btn.clicked.connect(self.start_section_profile)
        self.region_stats_btn.clicked.connect(self.start_region_statistics)
        self.distance_measure_btn.clicked.connect(self.start_distance_measurement)
        self.angle_measure_btn.clicked.connect(self.start_angle_measurement)
        
        # 导出功能
        self.export_btn.clicked.connect(self.export_depth_map)
    
    def on_single_view_clicked(self):
        """切换到单视图"""
        self.single_view_btn.setChecked(True)
        self.multi_view_btn.setChecked(False)
        self.view_stack.setCurrentIndex(0)
        self.update_depth_map()
    
    def on_multi_view_clicked(self):
        """切换到多视图"""
        self.single_view_btn.setChecked(False)
        self.multi_view_btn.setChecked(True)
        self.view_stack.setCurrentIndex(1)
        self.update_multi_view()
    
    def update_depth_map(self):
        """更新深度图显示，支持标记线绘制"""
        # 清除当前图像
        self.ax_single.clear()
        
        # 应用对比度
        depth_data = np.clip(self.depth_image, self.contrast_min, self.contrast_max)
        
        # 获取色彩映射
        cmap = self.cmap_manager.get_colormap(self.current_colormap, self.colormap_reversed)
        
        # 绘制深度图，使用更高质量的插值，origin改为lower与标记线坐标系统保持一致
        self.im_single = self.ax_single.imshow(
            depth_data, 
            extent=self.extent, 
            origin='lower', 
            cmap=cmap, 
            interpolation='lanczos',  # 更高质量的插值方法
            aspect='auto'
        )
        
        # 绘制标记线
        if self.marker_lines_2d is not None:
            for line_data in self.marker_lines_2d:
                if 'points' in line_data:
                    points = np.array(line_data['points'])
                    color = line_data.get('color', (1, 0, 0))  # 默认红色
                    label = line_data.get('label', '')
                    
                    # 绘制线段
                    self.ax_single.plot(points[:, 0], points[:, 1], color=color, linewidth=2, alpha=0.8)
                    print(f"[DIALOG] 已绘制标记线: {label}, 颜色: {color}, 点数: {len(points)}")
                    
                    # 如果有标签，添加到线段中间位置
                    if label:
                        mid_idx = len(points) // 2
                        mid_point = points[mid_idx]
                        self.ax_single.text(mid_point[0], mid_point[1], label, fontsize=8, 
                                          color=color, backgroundcolor='white', alpha=0.7, 
                                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # 添加颜色条
        cbar = self.figure_single.colorbar(self.im_single, ax=self.ax_single, shrink=0.9)
        cbar.set_label('深度值 (mm)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # 添加标题和标签
        self.ax_single.set_title('投影深度图', fontsize=12, fontweight='bold')
        self.ax_single.set_xlabel('X (mm)', fontsize=10)
        self.ax_single.set_ylabel('Y (mm)', fontsize=10)
        
        # 刷新画布
        self.canvas_single.draw()
        
        # 优化网格线显示
        self.ax_single.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 更新显示
        self.figure_single.tight_layout()
        self.canvas_single.draw()
    
    def update_multi_view(self):
        """更新多视图显示"""
        if len(self.views) != 4:
            return
        
        # 应用对比度
        depth_data = np.clip(self.depth_image, self.contrast_min, self.contrast_max)
        
        # 获取色彩映射
        cmap = self.cmap_manager.get_colormap(self.current_colormap, self.colormap_reversed)
        
        # 1. 原始深度图
        self.views[0].clear()
        self.views[0].imshow(depth_data, extent=self.extent, origin='upper', 
                           cmap=cmap, interpolation='bicubic')
        self.views[0].set_title('原始深度图')
        self.views[0].set_xlabel('X (mm)')
        self.views[0].set_ylabel('Y (mm)')
        
        # 2. 增强后深度图（添加边缘增强）
        self.views[1].clear()
        # 简单的边缘增强
        from scipy.ndimage import sobel
        edges = sobel(depth_data)
        enhanced = depth_data + 0.1 * edges
        self.views[1].imshow(enhanced, extent=self.extent, origin='upper', 
                           cmap=cmap, interpolation='bicubic')
        self.views[1].set_title('增强后深度图')
        self.views[1].set_xlabel('X (mm)')
        self.views[1].set_ylabel('Y (mm)')
        
        # 3. 3D表面图
        self.views[2].clear()
        x = np.linspace(self.extent[0], self.extent[1], depth_data.shape[1])
        y = np.linspace(self.extent[2], self.extent[3], depth_data.shape[0])
        X, Y = np.meshgrid(x, y)
        self.views[2].contourf(X, Y, depth_data, 20, cmap=cmap)
        self.views[2].set_title('等高线图')
        self.views[2].set_xlabel('X (mm)')
        self.views[2].set_ylabel('Y (mm)')
        
        # 4. 统计直方图
        self.views[3].clear()
        self.views[3].hist(depth_data.flatten(), bins=50, alpha=0.7)
        self.views[3].set_title('深度值分布')
        self.views[3].set_xlabel('深度值 (mm)')
        self.views[3].set_ylabel('频率')
        
        # 更新所有画布
        for canvas in self.canvases:
            canvas.figure.tight_layout()
            canvas.draw()
    
    def adjust_contrast(self):
        """调整对比度"""
        self.contrast_min = self.contrast_min_slider.value() / 100.0
        self.contrast_max = self.contrast_max_slider.value() / 100.0
        
        # 更新显示
        if self.view_stack.currentIndex() == 0:
            self.update_depth_map()
        else:
            self.update_multi_view()
    
    def change_colormap(self):
        """更改色彩映射"""
        self.current_colormap = self.colormap_combo.currentText()
        self.colormap_reversed = self.colormap_reverse_cb.isChecked()
        
        # 更新显示
        if self.view_stack.currentIndex() == 0:
            self.update_depth_map()
        else:
            self.update_multi_view()
    
    def apply_auto_contrast(self):
        """应用自动对比度"""
        # 使用更智能的自动对比度算法
        valid_depth = self.depth_image[~np.isnan(self.depth_image)]
        if len(valid_depth) > 0:
            # 使用1%和99%百分位作为对比度范围，避免极端值影响
            self.contrast_min = np.percentile(valid_depth, 1)
            self.contrast_max = np.percentile(valid_depth, 99)
        else:
            self.contrast_min = np.min(self.depth_image)
            self.contrast_max = np.max(self.depth_image)
        
        # 更新滑块
        self.contrast_min_slider.setValue(int(self.contrast_min * 100))
        self.contrast_max_slider.setValue(int(self.contrast_max * 100))
        
        # 立即应用对比度更新
        if self.view_stack.currentIndex() == 0:
            self.update_depth_map()
        else:
            self.update_multi_view()
        
        # 更新显示
        if self.view_stack.currentIndex() == 0:
            self.update_depth_map()
        else:
            self.update_multi_view()
    
    def start_section_profile(self):
        """开始剖面线分析"""
        # 设置标记模式
        self.section_profile_mode = "select_start"
        self.section_profile_points = []
        self.section_profile_btn.setText("选择剖面线起点")
        self.section_profile_view.show()
        
        # 连接鼠标点击事件
        if hasattr(self, 'ax_single'):
            self.single_view_cid = self.ax_single.figure.canvas.mpl_connect('button_press_event', self.on_section_profile_click)
        elif hasattr(self, 'multi_view_axes'):
            for ax in self.multi_view_axes.values():
                cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_section_profile_click)
                if not hasattr(self, 'multi_view_cids'):
                    self.multi_view_cids = []
                self.multi_view_cids.append(cid)
        
        QMessageBox.information(self, "剖面线分析", "请在深度图上点击选择剖面线的两个端点")
    
    def start_region_statistics(self):
        """开始区域统计"""
        # 简单实现：显示整个深度图的统计信息
        stats = self.analyzer.calculate_region_statistics(
            self.extent[0], self.extent[2], self.extent[1], self.extent[3]
        )
        
        stats_text = f"""
        最小深度: {stats['min']:.3f} mm
        最大深度: {stats['max']:.3f} mm
        平均深度: {stats['mean']:.3f} mm
        深度标准差: {stats['std']:.3f} mm
        区域面积: {stats['area']:.2f} mm²
        采样点数: {stats['points']}
        """
        
        self.region_stats_display.setText(stats_text)
    
    def start_distance_measurement(self):
        """开始距离测量"""
        # 清除之前的测量
        self.clear_distance_measurement()
        
        # 设置测量模式
        self.measurement_mode = "distance"
        self.distance_points = []
        self.distance_btn.setText("选择距离测量起点")
        
        # 连接鼠标点击事件
        self.connect_measurement_events()
        
        QMessageBox.information(self, "距离测量", "请在深度图上点击选择两个点来测量距离")
    
    def start_angle_measurement(self):
        """开始角度测量"""
        # 清除之前的测量
        self.clear_angle_measurement()
        
        # 设置测量模式
        self.measurement_mode = "angle"
        self.angle_points = []
        self.angle_btn.setText("选择角度测量第一个点")
        
        # 连接鼠标点击事件
        self.connect_measurement_events()
        
        QMessageBox.information(self, "角度测量", "请在深度图上点击选择三个点来测量角度")
    
    def export_depth_map(self):
        """导出深度图"""
        # 获取当前显示的图像
        if self.view_stack.currentIndex() == 0:
            # 单视图
            current_figure = self.figure_single
        else:
            # 多视图
            current_figure = self.canvases[0].figure
        
        # 获取导出格式
        export_format = self.export_format_combo.currentText().lower()
        
        # 显示文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"导出深度图为{export_format.upper()}", 
            f"depth_map.{export_format}", 
            f"{export_format.upper()} Files (*.{export_format});;All Files (*.*)"
        )
        
        if file_path:
            try:
                if export_format == "raw":
                    # 导出原始深度数据
                    np.save(file_path, self.depth_image)
                    QMessageBox.information(self, "导出成功", f"原始深度数据已保存到：{file_path}")
                else:
                    # 导出图像
                    current_figure.savefig(
                        file_path, 
                        format=export_format, 
                        dpi=300, 
                        bbox_inches='tight', 
                        pad_inches=0.1
                    )
                    QMessageBox.information(self, "导出成功", f"深度图已保存到：{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出过程中发生错误：{str(e)}")
    
    def on_section_profile_click(self, event):
        """处理剖面线点击事件"""
        if not hasattr(self, 'section_profile_mode') or self.section_profile_mode not in ["select_start", "select_end"]:
            return
            
        if event.inaxes:
            # 获取点击位置的坐标
            x, y = event.xdata, event.ydata
            
            if self.section_profile_mode == "select_start":
                # 保存起点
                self.section_profile_points.append((x, y))
                self.section_profile_btn.setText("选择剖面线终点")
                self.section_profile_mode = "select_end"
                
                # 在图像上标记起点
                if hasattr(self, 'ax_single'):
                    self.start_marker, = self.ax_single.plot(x, y, 'ro', markersize=8, label='起点')
                    self.ax_single.figure.canvas.draw()
                elif hasattr(self, 'multi_view_axes') and 'depth' in self.multi_view_axes:
                    self.start_marker, = self.multi_view_axes['depth'].plot(x, y, 'ro', markersize=8, label='起点')
                    self.multi_view_axes['depth'].figure.canvas.draw()
                    
            elif self.section_profile_mode == "select_end":
                # 保存终点
                self.section_profile_points.append((x, y))
                self.section_profile_btn.setText("开始剖面线分析")
                self.section_profile_mode = "completed"
                
                # 在图像上标记终点和剖面线
                if hasattr(self, 'ax_single'):
                    self.end_marker, = self.ax_single.plot(x, y, 'go', markersize=8, label='终点')
                    self.section_line, = self.ax_single.plot(
                        [self.section_profile_points[0][0], x], 
                        [self.section_profile_points[0][1], y], 
                        'r-', linewidth=2, label='剖面线'
                    )
                    self.ax_single.figure.canvas.draw()
                elif hasattr(self, 'multi_view_axes') and 'depth' in self.multi_view_axes:
                    self.end_marker, = self.multi_view_axes['depth'].plot(x, y, 'go', markersize=8, label='终点')
                    self.section_line, = self.multi_view_axes['depth'].plot(
                        [self.section_profile_points[0][0], x], 
                        [self.section_profile_points[0][1], y], 
                        'r-', linewidth=2, label='剖面线'
                    )
                    self.multi_view_axes['depth'].figure.canvas.draw()
                
                # 计算剖面线深度分布
                self.calculate_and_show_section_profile()
                
                # 断开鼠标点击事件
                self.disconnect_section_profile_events()
                
    def calculate_and_show_section_profile(self):
        """计算并显示剖面线的深度分布"""
        if len(self.section_profile_points) != 2:
            return
            
        x1, y1 = self.section_profile_points[0]
        x2, y2 = self.section_profile_points[1]
        
        # 计算剖面线深度分布
        distances, depth_values = self.analyzer.calculate_section_profile(x1, y1, x2, y2, num_points=200)
        
        # 创建图表显示深度分布
        self.section_profile_view.setLayout(QVBoxLayout())
        
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(distances, depth_values, 'b-', linewidth=2)
        ax.set_title('剖面线深度分布')
        ax.set_xlabel('距离 (mm)')
        ax.set_ylabel('深度 (mm)')
        ax.grid(True)
        
        # 添加统计信息
        stats_text = f"最小深度: {np.min(depth_values):.3f} mm\n"
        stats_text += f"最大深度: {np.max(depth_values):.3f} mm\n"
        stats_text += f"平均深度: {np.mean(depth_values):.3f} mm\n"
        stats_text += f"深度范围: {np.max(depth_values) - np.min(depth_values):.3f} mm"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        canvas = FigureCanvas(fig)
        self.section_profile_view.layout().addWidget(canvas)
        
        # 添加清除按钮
        clear_btn = QPushButton("清除剖面线")
        clear_btn.clicked.connect(self.clear_section_profile)
        self.section_profile_view.layout().addWidget(clear_btn)
        
    def disconnect_section_profile_events(self):
        """断开剖面线相关的事件连接"""
        if hasattr(self, 'single_view_cid'):
            if hasattr(self, 'ax_single'):
                self.ax_single.figure.canvas.mpl_disconnect(self.single_view_cid)
            delattr(self, 'single_view_cid')
        
        if hasattr(self, 'multi_view_cids'):
            for cid in self.multi_view_cids:
                if hasattr(self, 'multi_view_axes'):
                    for ax in self.multi_view_axes.values():
                        ax.figure.canvas.mpl_disconnect(cid)
            delattr(self, 'multi_view_cids')
            
    def clear_section_profile(self):
        """清除剖面线分析"""
        # 断开事件连接
        self.disconnect_section_profile_events()
        
        # 清除标记
        if hasattr(self, 'start_marker'):
            self.start_marker.remove()
            delattr(self, 'start_marker')
        if hasattr(self, 'end_marker'):
            self.end_marker.remove()
            delattr(self, 'end_marker')
        if hasattr(self, 'section_line'):
            self.section_line.remove()
            delattr(self, 'section_line')
        
        # 清除视图
        self.section_profile_view.hide()
        self.section_profile_view.setLayout(None)
        
        # 重置状态
        self.section_profile_mode = None
        self.section_profile_points = []
        self.section_profile_btn.setText("开始剖面线分析")
        
        # 重绘图像
        if hasattr(self, 'ax_single'):
            self.ax_single.figure.canvas.draw()
        elif hasattr(self, 'multi_view_axes'):
            for ax in self.multi_view_axes.values():
                ax.figure.canvas.draw()
    
    def connect_measurement_events(self):
        """连接测量相关的事件"""
        # 断开之前的事件连接
        self.disconnect_measurement_events()
        
        # 连接新的事件
        if hasattr(self, 'ax_single'):
            self.measurement_cid = self.ax_single.figure.canvas.mpl_connect('button_press_event', self.on_measurement_click)
        elif hasattr(self, 'multi_view_axes'):
            self.measurement_cids = []
            for ax in self.multi_view_axes.values():
                cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_measurement_click)
                self.measurement_cids.append(cid)
    
    def disconnect_measurement_events(self):
        """断开测量相关的事件"""
        if hasattr(self, 'measurement_cid'):
            if hasattr(self, 'ax_single'):
                self.ax_single.figure.canvas.mpl_disconnect(self.measurement_cid)
            delattr(self, 'measurement_cid')
        
        if hasattr(self, 'measurement_cids'):
            if hasattr(self, 'multi_view_axes'):
                for i, ax in enumerate(self.multi_view_axes.values()):
                    if i < len(self.measurement_cids):
                        ax.figure.canvas.mpl_disconnect(self.measurement_cids[i])
            delattr(self, 'measurement_cids')
    
    def on_measurement_click(self, event):
        """处理测量点击事件"""
        if not hasattr(self, 'measurement_mode'):
            return
            
        if event.inaxes:
            # 获取点击位置的坐标
            x, y = event.xdata, event.ydata
            
            if self.measurement_mode == "distance":
                self.handle_distance_click(event)
            elif self.measurement_mode == "angle":
                self.handle_angle_click(event)
    
    def handle_distance_click(self, event):
        """处理距离测量点击"""
        x, y = event.xdata, event.ydata
        self.distance_points.append((x, y))
        
        # 确定当前坐标轴
        ax = event.inaxes
        
        if len(self.distance_points) == 1:
            # 显示第一个点
            self.distance_marker1, = ax.plot(x, y, 'ro', markersize=8, label='起点')
            ax.figure.canvas.draw()
            self.distance_btn.setText("选择距离测量终点")
        elif len(self.distance_points) == 2:
            # 显示第二个点和连接线
            self.distance_marker2, = ax.plot(x, y, 'go', markersize=8, label='终点')
            self.distance_line, = ax.plot(
                [self.distance_points[0][0], x], 
                [self.distance_points[0][1], y], 
                'r-', linewidth=2
            )
            
            # 计算距离
            distance = np.sqrt((self.distance_points[1][0] - self.distance_points[0][0])**2 + 
                              (self.distance_points[1][1] - self.distance_points[0][1])**2)
            
            # 显示距离值
            mid_x = (self.distance_points[0][0] + self.distance_points[1][0]) / 2
            mid_y = (self.distance_points[0][1] + self.distance_points[1][1]) / 2
            self.distance_text = ax.text(mid_x, mid_y, f"{distance:.3f} mm", 
                                        ha='center', va='bottom', 
                                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.figure.canvas.draw()
            
            # 重置按钮
            self.distance_btn.setText("开始距离测量")
            
            # 断开事件连接
            self.disconnect_measurement_events()
            
            # 显示测量结果
            QMessageBox.information(self, "距离测量结果", f"测量距离: {distance:.3f} mm")
    
    def handle_angle_click(self, event):
        """处理角度测量点击"""
        x, y = event.xdata, event.ydata
        self.angle_points.append((x, y))
        
        # 确定当前坐标轴
        ax = event.inaxes
        
        if len(self.angle_points) == 1:
            # 显示第一个点
            self.angle_marker1, = ax.plot(x, y, 'ro', markersize=8, label='点1')
            ax.figure.canvas.draw()
            self.angle_btn.setText("选择角度测量第二个点")
        elif len(self.angle_points) == 2:
            # 显示第二个点
            self.angle_marker2, = ax.plot(x, y, 'go', markersize=8, label='点2')
            ax.figure.canvas.draw()
            self.angle_btn.setText("选择角度测量第三个点")
        elif len(self.angle_points) == 3:
            # 显示第三个点
            self.angle_marker3, = ax.plot(x, y, 'bo', markersize=8, label='点3')
            
            # 绘制角度线
            self.angle_line1, = ax.plot(
                [self.angle_points[1][0], self.angle_points[0][0]], 
                [self.angle_points[1][1], self.angle_points[0][1]], 
                'r-', linewidth=2
            )
            self.angle_line2, = ax.plot(
                [self.angle_points[1][0], self.angle_points[2][0]], 
                [self.angle_points[1][1], self.angle_points[2][1]], 
                'g-', linewidth=2
            )
            
            # 计算角度
            v1 = np.array([self.angle_points[0][0] - self.angle_points[1][0], 
                         self.angle_points[0][1] - self.angle_points[1][1]])
            v2 = np.array([self.angle_points[2][0] - self.angle_points[1][0], 
                         self.angle_points[2][1] - self.angle_points[1][1]])
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 * norm_v2 > 0:
                cos_angle = dot_product / (norm_v1 * norm_v2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # 确保在有效范围内
                angle = np.arccos(cos_angle) * 180 / np.pi
            else:
                angle = 0.0
            
            # 显示角度值
            self.angle_text = ax.text(self.angle_points[1][0], self.angle_points[1][1], 
                                    f"{angle:.2f}°", ha='center', va='top', 
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.figure.canvas.draw()
            
            # 重置按钮
            self.angle_btn.setText("开始角度测量")
            
            # 断开事件连接
            self.disconnect_measurement_events()
            
            # 显示测量结果
            QMessageBox.information(self, "角度测量结果", f"测量角度: {angle:.2f}°")
    
    def clear_distance_measurement(self):
        """清除距离测量"""
        # 清除标记和线条
        if hasattr(self, 'distance_marker1'):
            self.distance_marker1.remove()
            delattr(self, 'distance_marker1')
        if hasattr(self, 'distance_marker2'):
            self.distance_marker2.remove()
            delattr(self, 'distance_marker2')
        if hasattr(self, 'distance_line'):
            self.distance_line.remove()
            delattr(self, 'distance_line')
        if hasattr(self, 'distance_text'):
            self.distance_text.remove()
            delattr(self, 'distance_text')
        
        # 重置状态
        self.distance_points = []
        if hasattr(self, 'distance_btn'):
            self.distance_btn.setText("开始距离测量")
        
        # 重绘图像
        if hasattr(self, 'ax_single'):
            self.ax_single.figure.canvas.draw()
        elif hasattr(self, 'multi_view_axes'):
            for ax in self.multi_view_axes.values():
                ax.figure.canvas.draw()
    
    def clear_angle_measurement(self):
        """清除角度测量"""
        # 清除标记和线条
        if hasattr(self, 'angle_marker1'):
            self.angle_marker1.remove()
            delattr(self, 'angle_marker1')
        if hasattr(self, 'angle_marker2'):
            self.angle_marker2.remove()
            delattr(self, 'angle_marker2')
        if hasattr(self, 'angle_marker3'):
            self.angle_marker3.remove()
            delattr(self, 'angle_marker3')
        if hasattr(self, 'angle_line1'):
            self.angle_line1.remove()
            delattr(self, 'angle_line1')
        if hasattr(self, 'angle_line2'):
            self.angle_line2.remove()
            delattr(self, 'angle_line2')
        if hasattr(self, 'angle_text'):
            self.angle_text.remove()
            delattr(self, 'angle_text')
        
        # 重置状态
        self.angle_points = []
        if hasattr(self, 'angle_btn'):
            self.angle_btn.setText("开始角度测量")
        
        # 重绘图像
        if hasattr(self, 'ax_single'):
            self.ax_single.figure.canvas.draw()
        elif hasattr(self, 'multi_view_axes'):
            for ax in self.multi_view_axes.values():
                ax.figure.canvas.draw()
    
    def disconnect_measurement_events(self):
        """断开测量相关的事件"""
        if hasattr(self, 'measurement_cid'):
            if hasattr(self, 'ax_single'):
                self.ax_single.figure.canvas.mpl_disconnect(self.measurement_cid)
            delattr(self, 'measurement_cid')
        
        if hasattr(self, 'measurement_cids'):
            if hasattr(self, 'multi_view_axes'):
                for cid in self.measurement_cids:
                    for ax in self.multi_view_axes.values():
                        ax.figure.canvas.mpl_disconnect(cid)
            delattr(self, 'measurement_cids')


class DepthImageDialog(QWidget):
    """显示投影深度图的对话框"""
    def __init__(self, depth_image, extent, title="投影深度图", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 900, 700)
        
        # 创建matplotlib画布 - 提高DPI和质量
        self.figure = Figure(figsize=(8, 6), dpi=120)
        self.canvas = FigureCanvas(self.figure)
        
        # 导入导航工具栏
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        
        # 布局
        layout = QVBoxLayout(self)
        
        # 添加导航工具栏
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        
        # 添加画布
        layout.addWidget(self.canvas)
        
        # 绘制深度图
        self.axes = self.figure.add_subplot(111)
        
        # 计算深度值统计信息
        depth_min = np.min(depth_image)
        depth_max = np.max(depth_image)
        depth_mean = np.mean(depth_image)
        depth_std = np.std(depth_image)
        
        # 绘制深度图 - 使用bicubic插值提高图像质量
        im = self.axes.imshow(depth_image, cmap='viridis', extent=extent, origin='upper', interpolation='bicubic')
        
        # 设置标签 - 改进字体和样式
        self.axes.set_xlabel('X坐标 (mm)', fontsize=12, fontweight='bold')
        self.axes.set_ylabel('Y坐标 (mm)', fontsize=12, fontweight='bold')
        self.axes.set_title('上颌投影深度图', fontsize=14, fontweight='bold')
        
        # 改进网格线
        self.axes.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
        
        # 添加颜色条 - 只绘制一次
        cbar = self.figure.colorbar(im, ax=self.axes, label='深度值 (mm)', pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
        # 添加详细的统计信息
        stats_text = f"最小深度: {depth_min:.3f} mm\n" \
                   f"最大深度: {depth_max:.3f} mm\n" \
                   f"平均深度: {depth_mean:.3f} mm\n" \
                   f"深度标准差: {depth_std:.3f} mm"
        
        # 改进统计信息显示样式
        self.axes.text(0.02, 0.98, stats_text, transform=self.axes.transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                     fontsize=10, fontweight='normal')
        
        # 优化布局
        self.figure.tight_layout(pad=2.0)
        
        self.canvas.draw()

class ProcessingThread(QThread):
    """处理线程，避免UI冻结（增强版，防止闪退）"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = True
        logger.debug(f"创建处理线程: {function.__name__}，参数数量: {len(args)}")
        
    def run(self):
        try:
            logger.debug(f"开始执行函数: {self.function.__name__}")
            # 捕获函数执行结果
            result = self.function(*self.args, **self.kwargs)
            
            # 检查是否停止标志
            if not self.is_running:
                logger.debug("线程已被停止")
                return
            
            # 检查结果是否有效
            if result is None:
                logger.warning("函数返回None")
                self.error_occurred.emit("处理函数返回空结果")
                return
            
            # 通用结果有效性检查
            if isinstance(result, tuple) and len(result) >= 2 and result[0] is None and result[1] is None:
                self.error_occurred.emit("处理失败: 返回空结果")
                return
            
            logger.debug(f"函数执行完成，结果类型: {type(result)}")
            self.processing_finished.emit(result)
        except Exception as e:
            if not self.is_running:
                logger.debug("线程已被停止，忽略异常")
                return
            
            logger.error(f"线程执行异常: {str(e)}", exc_info=True)
            
            # 构建更友好的错误信息
            error_msg = str(e)

            if "models_dict must contain" in error_msg or "models_dict缺少必要的键" in error_msg:
                error_msg = "模型数据不完整: 请确保已正确加载上下颌模型"
            elif "not enough values to unpack" in error_msg:
                error_msg = "返回值解析错误: 无法解析平面生成结果"
            
            self.error_occurred.emit(error_msg)
    
    def stop(self, timeout=3000):
        """停止线程，带超时机制
        
        Args:
            timeout: 等待线程停止的超时时间（毫秒），默认为3000毫秒
            
        Returns:
            bool: 线程是否成功停止
        """
        logger.debug(f"停止处理线程，超时设置: {timeout}毫秒")
        
        # 设置停止标志
        self.is_running = False
        
        # 等待线程终止，但设置超时
        if self.isRunning():
            success = self.wait(timeout)
            if not success:
                logger.warning("线程停止超时，可能存在资源泄漏")
                return False
        
        logger.debug("线程已成功停止")
        return True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_data()
        self.update_status()
        # 初始化线程管理器，使用线程池机制
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # 限制最大线程数为4
        print(f"[MAIN_WINDOW] 线程池初始化完成，最大线程数: {self.thread_pool.maxThreadCount()}")
        
        # 初始化实时预览生成器
        self.preview_generator = DepthPreviewGenerator()
        
        # 初始化投影深度生成器
        self.depth_generator = ProjectionDepthGenerator(self.viewer)
        
        # 添加定时器，定期打印信息，降低频率以减少资源消耗
        self.print_timer = QTimer()
        self.print_timer.timeout.connect(self._print_debug_info)
        self.print_timer.start(5000)  # 每5秒打印一次
        
        # 添加UI响应性监控定时器
        self.ui_watchdog_timer = QTimer()
        self.ui_watchdog_timer.timeout.connect(self._check_ui_responsiveness)
        self.ui_watchdog_timer.start(1000)  # 每秒检查一次UI响应性
    
    def _print_debug_info(self):
        """定期打印调试信息"""
        print(f"[DEBUG] 主窗口仍在运行，可见性: {self.isVisible()}")
        if hasattr(self, 'viewer'):
            print(f"[DEBUG] 3D查看器存在，类型: {type(self.viewer)}")
        if hasattr(self, 'models'):
            print(f"[DEBUG] 模型数量: {len(self.models)}")
        if hasattr(self, 'thread_pool'):
            print(f"[DEBUG] 线程池活跃线程数: {self.thread_pool.activeThreadCount()}")
    
    def _check_ui_responsiveness(self):
        """检查UI响应性，确保界面流畅"""
        try:
            # 简单的UI响应性检查：更新状态标签的时间戳
            current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
            # 只在空闲时更新，避免影响性能
            if self.thread_pool.activeThreadCount() == 0:
                self.status_label.setToolTip(f"最后响应时间: {current_time}")
        except Exception as e:
            print(f"[CRITICAL WARNING] UI响应性检查失败: {e}")
        
    def init_ui(self):
        """初始化用户界面"""
        print("初始化主窗口界面")
        
        # 设置主窗口属性
        self.setWindowTitle("牙列标记系统")
        self.setGeometry(100, 100, 1600, 900)
        self.is_running = True  # 标记应用是否正在运行
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        # 使用QScrollArea包装控制面板，确保内容过多时可以滚动
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        splitter.addWidget(scroll_area)
        
        # 右侧3D视图
        self.viewer = ModelViewer(self)
        # 连接信号
        self.viewer.model_loaded.connect(self.on_model_loaded)
        self.viewer.model_error.connect(self.on_model_error)
        
        # 连接平面拟合完成信号
        self.viewer.fit_plane_completed.connect(self.on_fit_plane_completed)
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.on_marked_points_updated)
        splitter.addWidget(self.viewer)
        
        # 设置分割比例和拉伸因子
        splitter.setSizes([400, 1200])
        # 设置拉伸因子，确保控制面板和3D视图按比例调整
        splitter.setStretchFactor(0, 1)  # 控制面板拉伸因子为1
        splitter.setStretchFactor(1, 3)  # 3D视图拉伸因子为3
        main_layout.addWidget(splitter)
        
        print("主窗口界面初始化完成")
        

    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        # 移除最大宽度限制，让控制面板可以根据窗口大小自动调整
        # panel.setMaximumWidth(400)
        panel.setStyleSheet("background-color: #f5f5f5;")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("牙列标记系统")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title)
        
        # 文件操作组
        file_group = QGroupBox("模型导入")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(5)
        
        self.maxilla_btn = QPushButton("导入上颌模型")
        self.maxilla_btn.clicked.connect(lambda: self.load_model("maxilla"))
        self.maxilla_btn.setStyleSheet("padding: 8px;")
        file_layout.addWidget(self.maxilla_btn)
        
        self.mandible_btn = QPushButton("导入下颌模型")
        self.mandible_btn.clicked.connect(lambda: self.load_model("mandible"))
        self.mandible_btn.setStyleSheet("padding: 8px;")
        file_layout.addWidget(self.mandible_btn)
        
        self.occlusion_btn = QPushButton("导入咬合关系模型")
        self.occlusion_btn.clicked.connect(lambda: self.load_model("occlusion"))
        self.occlusion_btn.setStyleSheet("padding: 8px;")
        file_layout.addWidget(self.occlusion_btn)
        
        # 显示文件路径的标签
        self.maxilla_path = QLabel("未加载")
        self.maxilla_path.setStyleSheet("font-size: 8pt; color: #666;")
        file_layout.addWidget(self.maxilla_path)
        
        self.mandible_path = QLabel("未加载")
        self.mandible_path.setStyleSheet("font-size: 8pt; color: #666;")
        file_layout.addWidget(self.mandible_path)
        
        self.occlusion_path = QLabel("未加载")
        self.occlusion_path.setStyleSheet("font-size: 8pt; color: #666;")
        file_layout.addWidget(self.occlusion_path)
        
        layout.addWidget(file_group)
        
        # 模型显示控制组
        display_group = QGroupBox("显示控制")
        display_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(5)
        
        # 模型可见性控制
        visibility_layout = QHBoxLayout()
        visibility_layout.setSpacing(5)
        self.show_maxilla = QCheckBox("上颌")
        self.show_maxilla.setChecked(True)
        self.show_maxilla.toggled.connect(self.toggle_model_visibility)
        visibility_layout.addWidget(self.show_maxilla)
        
        self.show_mandible = QCheckBox("下颌")
        self.show_mandible.setChecked(True)
        self.show_mandible.toggled.connect(self.toggle_model_visibility)
        visibility_layout.addWidget(self.show_mandible)
        
        # 添加咬合关系模型显示控制
        self.show_occlusion = QCheckBox("咬合关系")
        self.show_occlusion.setChecked(True)
        self.show_occlusion.toggled.connect(self.toggle_model_visibility)
        visibility_layout.addWidget(self.show_occlusion)
        
        display_layout.addLayout(visibility_layout)
        
        # 透明度控制
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(QLabel("透明度:"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(30)
        self.transparency_slider.valueChanged.connect(self.update_transparency)
        transparency_layout.addWidget(self.transparency_slider)
        
        self.transparency_value = QLabel("30%")
        transparency_layout.addWidget(self.transparency_value)
        
        display_layout.addLayout(transparency_layout)
        
        # 视图控制按钮
        view_control_layout = QHBoxLayout()
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setStyleSheet("padding: 5px;")
        view_control_layout.addWidget(self.reset_view_btn)
        
        self.screenshot_btn = QPushButton("导出截图")
        self.screenshot_btn.clicked.connect(self.show_screenshot_options)
        self.screenshot_btn.setStyleSheet("padding: 5px;")
        view_control_layout.addWidget(self.screenshot_btn)
        
        display_layout.addLayout(view_control_layout)
        

        
        # 移除平面法线显示开关
        
        # 添加视图控制组件
        self._add_view_controls(layout)
        
        layout.addWidget(display_group)
        

        
        # 创建并添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        

        

        
        # 创建投影功能面板
        projection_group = QGroupBox("投影功能")
        projection_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; padding: 12px; width: 100%; }")
        projection_layout = QVBoxLayout(projection_group)
        projection_layout.setSpacing(12)  # 调整间距，提高可读性
        projection_layout.setContentsMargins(12, 12, 12, 12)
        projection_layout.addStretch(0)
        
        # 标记功能子组
        marking_subgroup = QGroupBox("标记功能")
        marking_subgroup.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 3px; margin-top: 5px; padding: 12px; width: 100%; }")
        marking_subgroup_layout = QVBoxLayout(marking_subgroup)
        marking_subgroup_layout.setSpacing(12)  # 调整间距，减少按钮重叠风险
        marking_subgroup_layout.setContentsMargins(8, 8, 8, 8)
        marking_subgroup_layout.addStretch(0)
        
        # 按钮容器和样式设置函数
        def create_styled_button(text, color, hover_color, pressed_color, clicked_slot=None, enabled=True):
            button = QPushButton(text)
            button.setFixedHeight(38)  # 进一步减小按钮高度，减少垂直空间占用
            button.setMinimumWidth(180)  # 减小最小宽度，提高灵活性
            # 移除最大宽度限制，让按钮可以根据容器宽度自动调整
            # button.setMaximumWidth(280)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 尺寸策略，允许水平扩展
            button.setAutoFillBackground(True)
            button.setFont(QFont("Microsoft YaHei", 9, QFont.Medium))  # 减小字体大小，适配更小的按钮
            button.setCursor(Qt.PointingHandCursor)  # 设置鼠标悬停时的光标样式
            button.setToolTip(text)  # 添加工具提示
            
            # 现代化样式表，使用PyQt5支持的CSS属性
            disabled_style = "QPushButton:disabled { background-color: #E0E0E0; color: #9E9E9E; border: 1px solid #BDBDBD; border-radius: 6px; }"
            base_style = f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; padding: 10px; border-radius: 6px; border: none; }}"
            hover_style = f"QPushButton:hover:enabled {{ background-color: {hover_color}; border: 1px solid {hover_color}; }}"
            pressed_style = f"QPushButton:pressed:enabled {{ background-color: {pressed_color}; border: 1px solid {pressed_color}; padding: 11px 9px 9px 11px; }}"
            
            button.setStyleSheet(f"{base_style} {hover_style} {pressed_style} {disabled_style}")
            
            if clicked_slot:
                button.clicked.connect(clicked_slot)
            
            button.setEnabled(enabled)
            
            return button
        
        # 启用标记平面功能按钮
        self.enable_marking_btn = create_styled_button(
            "启用标记平面功能", "#2196F3", "#1976D2", "#1565C0", self.enable_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.enable_marking_btn)
        
        # 添加分组分隔线
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("background-color: #E0E0E0; margin: 8px 0;")
        marking_subgroup_layout.addWidget(separator1)
        
        # 标记牙颌按钮
        self.enable_maxilla_marking_btn = create_styled_button(
            "标记牙颌", "#4CAF50", "#45a049", "#3d8b40", self.enable_maxilla_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.enable_maxilla_marking_btn)
        
        # 添加分组分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("background-color: #E0E0E0; margin: 8px 0;")
        marking_subgroup_layout.addWidget(separator2)
        
        # 启用上颌牙列多点标记功能按钮
        self.enable_maxilla_alveolar_ridge_btn = create_styled_button(
            "标记上颌牙列", "#FF5722", "#F4511E", "#E64A19", self.enable_maxilla_alveolar_ridge_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.enable_maxilla_alveolar_ridge_btn)
        
        # 启用下颌牙列多点标记功能按钮
        self.enable_mandible_alveolar_ridge_btn = create_styled_button(
            "标记下颌牙列", "#795548", "#6D4C41", "#5D4037", self.enable_mandible_alveolar_ridge_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.enable_mandible_alveolar_ridge_btn)
        
        # 添加分组分隔线
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("background-color: #E0E0E0; margin: 8px 0;")
        marking_subgroup_layout.addWidget(separator3)
        
        # 标记前端按钮
        self.divide_maxilla_btn = create_styled_button(
            "标记前端", "#00BCD4", "#00ACC1", "#0097A7", self.enable_divide_maxilla_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.divide_maxilla_btn)
        
        # 标记后端按钮
        self.enable_mandible_crest_btn = create_styled_button(
            "标记后端", "#FF9800", "#F57C00", "#E65100", self.enable_divide_mandible_marking
        )
        # 移除居中对齐，让按钮可以水平扩展
        marking_subgroup_layout.addWidget(self.enable_mandible_crest_btn)
        

        
        projection_layout.addWidget(marking_subgroup)
        
        # 处理功能子组
        processing_subgroup = QGroupBox("处理功能")
        processing_subgroup.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 3px; margin-top: 5px; padding: 8px; width: 100%; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }")
        processing_subgroup_layout = QVBoxLayout(processing_subgroup)
        processing_subgroup_layout.setSpacing(10)
        processing_subgroup_layout.setContentsMargins(0, 5, 0, 5)
        
        # 生成上颌投影图像按钮（合并所有功能）
        self.produce_projection_btn = create_styled_button(
            "生成上颌投影图像", "#4CAF50", "#388E3C", "#2E7D32", self.produce_projection_image, enabled=False
        )
        # 移除居中对齐，让按钮可以水平扩展
        processing_subgroup_layout.addWidget(self.produce_projection_btn)
        
        # 生成下颌投影图像按钮（合并所有功能）
        self.generate_mandible_projection_btn = create_styled_button(
            "生成下颌投影图像", "#2196F3", "#1976D2", "#1565C0", self.generate_mandible_projection_image, enabled=False
        )
        # 移除居中对齐，让按钮可以水平扩展
        processing_subgroup_layout.addWidget(self.generate_mandible_projection_btn)
        
        # 保留原始按钮用于调试
        self.show_crest_line_btn = create_styled_button(
            "显示后槽牙槽嵴连线", "#FF9800", "#F57C00", "#E65100", self.show_mandible_crest_line, enabled=False
        )
        processing_subgroup_layout.addWidget(self.show_crest_line_btn, alignment=Qt.AlignCenter)
        self.show_crest_line_btn.hide()
        
        self.project_crest_btn = create_styled_button(
            "投影后槽牙槽嵴到平面", "#9C27B0", "#8E24AA", "#7B1FA2", self.project_mandible_crest_to_plane, enabled=False
        )
        processing_subgroup_layout.addWidget(self.project_crest_btn, alignment=Qt.AlignCenter)
        self.project_crest_btn.hide()
        
        self.generate_projection_btn = create_styled_button(
            "生成投影图像", "#2196F3", "#1976D2", "#1565C0", self.generate_projection, enabled=False
        )
        processing_subgroup_layout.addWidget(self.generate_projection_btn, alignment=Qt.AlignCenter)
        self.generate_projection_btn.hide()
        
        projection_layout.addWidget(processing_subgroup)
        
        # 清除标记点按钮
        self.clear_marks_btn = create_styled_button(
            "清除标记点", "#f44336", "#da190b", "#b21709", self.clear_marked_points
        )
        # 移除居中对齐，让按钮可以水平扩展
        projection_layout.addWidget(self.clear_marks_btn)
        
        layout.addWidget(projection_group)
        
        # 移除了深度图生成设置和优化控制面板
        
        # 清除按钮
        self.clear_btn = create_styled_button(
            "清除所有模型", "#e74c3c", "#c62828", "#b71c1c", self.clear_all_models
        )
        # 移除居中对齐，让按钮可以水平扩展
        layout.addWidget(self.clear_btn)
        
        # 添加拉伸空间
        layout.addStretch(1)
        
        return panel
    
    def init_data(self):
        """初始化数据"""
        print("初始化数据")
        
        # 模型数据
        self.models = {
            "maxilla": None,   # 上颌模型
            "mandible": None,  # 下颌模型
            "occlusion": None  # 咬合关系模型
        }
        

        
        # 中间结果存储
        self.intermediate_results = {}
        
        # 标记功能相关
        self.marked_points = []
        self.plane_params = None
        
        # 加载器和生成器
        self.model_loader = ModelLoader()
        
        # 线程控制
        self.current_thread = None
        
        # 用于深度图与3D模型联动的数据
        self.projection_data = {
            'points_2d': None,
            'projected_points_3d': None,
            'depth_values': None,
            'interpolation_coords': None,
            'depth_image': None,
            'depth_ax': None
        }
        
        print("数据初始化完成")
    
    def load_model(self, model_type):
        """加载模型文件
        
        Args:
            model_type: 模型类型 (maxilla, mandible, occlusion)
        """
        print(f"开始加载{model_type}模型")
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择{self.get_model_type_name(model_type)}模型",
            "",
            "STL文件 (*.stl);;所有文件 (*)"
        )
        
        if file_path:
            self.status_label.setText(f"正在加载{self.get_model_type_name(model_type)}模型...")
            self.progress_bar.setValue(30)
            
            try:
                # 使用线程加载模型
                self.current_thread = ProcessingThread(
                    self.model_loader.load_stl,
                    file_path
                )
                self.current_thread.processing_finished.connect(
                    lambda mesh, mt=model_type, path=file_path: self.on_model_loaded_thread(mesh, mt, path)
                )
                self.current_thread.error_occurred.connect(
                    lambda error, mt=model_type: self.on_model_error_thread(error, mt)
                )
                self.current_thread.start()
                
            except Exception as e:
                self._handle_error(f"加载模型时发生错误: {str(e)}", "critical", "error")
    
    def on_model_loaded_thread(self, mesh, model_type, file_path):
        """模型加载完成后的处理（线程回调）"""
        if mesh:
            # 保存模型
            self.models[model_type] = mesh
            
            # 更新UI
            if model_type == "maxilla":
                self.maxilla_path.setText(os.path.basename(file_path))
                color = [0.7, 0.7, 0.7]  # 灰色
            elif model_type == "mandible":
                self.mandible_path.setText(os.path.basename(file_path))
                color = [0.7, 0.7, 0.7]  # 灰色
            else:
                self.occlusion_path.setText(os.path.basename(file_path))
                color = [0.7, 0.7, 0.7]  # 灰色
            
            # 添加到查看器并检查结果
            success = self.viewer.add_model(
                model_type,
                mesh,
                color=color,
                model_type=model_type
            )
            
            self.progress_bar.setValue(100)
            
            if success:
                self.status_label.setText(f"{self.get_model_type_name(model_type)}模型加载成功并显示")
                # 更新按钮状态
                self.update_button_states()
            else:
                self.status_label.setText(f"{self.get_model_type_name(model_type)}模型加载成功但无法显示")
                self._handle_error(f"{self.get_model_type_name(model_type)}模型添加到查看器失败", "warning", "error")
        else:
            self._handle_error("无法加载模型文件", "warning", "error")
    
    def on_model_error_thread(self, error, model_type):
        """模型加载错误处理（线程回调）"""
        self._handle_error(f"加载{self.get_model_type_name(model_type)}模型时出错:\n{error}", "critical", "error")
    
    def on_model_loaded(self, model_name):
        """模型加载到查看器的回调"""
        print(f"模型{model_name}已添加到查看器")
    
    def on_model_error(self, error):
        """模型加载到查看器错误的回调"""
        print(f"查看器错误: {error}")
    
    def toggle_model_visibility(self):
        """切换模型可见性"""
        self.viewer.toggle_model_visibility("maxilla", self.show_maxilla.isChecked())
        self.viewer.toggle_model_visibility("mandible", self.show_mandible.isChecked())
        if self.models["occlusion"]:
            self.viewer.toggle_model_visibility("occlusion", self.show_occlusion.isChecked())
    
    def show_all_models(self):
        """显示所有三维模型"""
        # 只恢复主要模型的可见性，辅助模型保持原状态
        self.viewer.toggle_model_visibility("maxilla", self.show_maxilla.isChecked())
        self.viewer.toggle_model_visibility("mandible", self.show_mandible.isChecked())
        if self.models["occlusion"]:
            self.viewer.toggle_model_visibility("occlusion", self.show_occlusion.isChecked())
        print(f"[MAIN_WINDOW] 重新显示了主要三维模型")
    
    def update_transparency(self, value):
        """更新模型透明度"""
        self.transparency_value.setText(f"{value}%")
        self.viewer.update_transparency(value)
    

    

            
            
                
                

    
    def closeEvent(self, event):
        """窗口关闭事件处理，确保线程正确停止"""
        logger.info("应用程序正在关闭，清理资源...")
        
        # 停止定时器
        if hasattr(self, 'print_timer') and self.print_timer.isActive():
            self.print_timer.stop()
            print("[MAIN_WINDOW] 调试信息定时器已停止")
        
        if hasattr(self, 'ui_watchdog_timer') and self.ui_watchdog_timer.isActive():
            self.ui_watchdog_timer.stop()
            print("[MAIN_WINDOW] UI响应性监控定时器已停止")
        
        # 等待所有线程完成
        if hasattr(self, 'thread_pool'):
            logger.info(f"关闭前等待 {self.thread_pool.activeThreadCount()} 个活跃线程完成")
            self.thread_pool.waitForDone(2000)  # 等待2秒
            print(f"[MAIN_WINDOW] 线程池已关闭")
        
        # 标记应用为不再运行
        self.is_running = False
        
        # 确保基本清理完成后再关闭
        QThread.msleep(200)
        event.accept()
    

    
    def handle_threshold_layout(self, enabled):
        """处理阈值布局的启用状态"""
        if hasattr(self, 'threshold_layout'):
            self.threshold_layout.setEnabled(enabled)
            # 处理布局中的每个项目
            for i in range(self.threshold_layout.count()):
                item = self.threshold_layout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(enabled)
            logger.info(f"{'显示' if enabled else '隐藏'}接触阈值调整控件")
        else:
            logger.warning("threshold_layout属性不存在，无法处理阈值布局")
    
    def update_projection_button_state(self):
        """统一更新生成投影按钮的状态"""
        try:
            # 检查模型存在情况
            models = self.viewer.get_models()
            has_maxilla = "maxilla" in models
            has_mandible = "mandible" in models
            
            # 检查是否有拟合平面
            has_plane = hasattr(self.viewer, 'plane_params') and self.viewer.plane_params is not None
            
            # 根据用户需求：平面拟合完成后即可运行投影功能，无需等待标记点
            # 分别更新不同投影按钮的状态
            maxilla_projection_enabled = has_maxilla and has_plane
            mandible_projection_enabled = has_mandible and has_plane
            
            # 更新按钮状态
            self.generate_projection_btn.setEnabled(maxilla_projection_enabled)
            self.produce_projection_btn.setEnabled(maxilla_projection_enabled)
            self.generate_mandible_projection_btn.setEnabled(mandible_projection_enabled)
        except Exception as e:
            logger.error(f"更新投影按钮状态失败: {str(e)}")
    
    def _handle_error(self, message, error_type="error", log_level="error", show_dialog=True):
        """统一错误处理方法
        
        Args:
            message: 错误消息
            error_type: 错误类型 ("error", "warning", "critical", "info")
            log_level: 日志级别 ("debug", "info", "warning", "error", "critical")
            show_dialog: 是否显示对话框，默认为True
        """
        # 确保在主线程中处理UI更新
        if not self.isVisible() and error_type != "info":
            # 窗口不可见时，只记录日志，不显示对话框
            log_method = getattr(logger, log_level)
            log_method(message)
            return
        
        try:
            # 根据错误类型设置状态文本
            if error_type in ["error", "critical", "warning"]:
                self.status_label.setText(f"{error_type}：{message}")
            else:
                self.status_label.setText(f"{error_type}：{message}")
        except Exception as status_e:
            logger.warning(f"更新状态标签失败: {status_e}")
        
        try:
            # 重置进度条
            self.progress_bar.setValue(0)
        except Exception as progress_e:
            logger.warning(f"重置进度条失败: {progress_e}")
        
        # 记录日志
        log_method = getattr(logger, log_level)
        log_method(message)
        
        # 为Open3D深度图捕获失败错误添加详细提示
        display_message = message
        if "Open3D深度图捕获失败" in message:
            display_message = "Open3D深度图捕获失败\n\n"
            display_message += "可能的原因：\n"
            display_message += "1. OpenGL渲染问题\n"
            display_message += "2. 3D模型加载不完整\n"
            display_message += "3. 系统图形驱动兼容性问题\n\n"
            display_message += "解决建议：\n"
            display_message += "1. 尝试重启应用程序\n"
            display_message += "2. 更新显卡驱动\n"
            display_message += "3. 确保模型文件完整且格式正确\n"
            display_message += "4. 程序已自动切换到备用深度图生成方法"
        elif "渲染" in message or "OpenGL" in message:
            display_message = f"{message}\n\n"
            display_message += "可能的原因：\n"
            display_message += "1. 显卡驱动过旧\n"
            display_message += "2. OpenGL版本不兼容\n"
            display_message += "3. 模型复杂度过高\n\n"
            display_message += "解决建议：\n"
            display_message += "1. 更新显卡驱动\n"
            display_message += "2. 降低模型复杂度\n"
            display_message += "3. 关闭其他占用GPU的程序\n"
        
        # 显示消息框
        if show_dialog:
            try:
                # 使用QApplication.processEvents()确保UI响应
                QApplication.processEvents()
                
                if error_type == "critical":
                    QMessageBox.critical(self, "错误", display_message)
                elif error_type == "warning":
                    QMessageBox.warning(self, "警告", display_message)
                elif error_type == "info":
                    QMessageBox.information(self, "提示", display_message)
                else:
                    QMessageBox.critical(self, "错误", display_message)
            except Exception as msg_e:
                logger.warning(f"显示错误消息失败: {msg_e}")
                # 尝试使用备用方式显示错误消息
                try:
                    print(f"[{error_type.upper()}] {message}")
                except:
                    pass
        
        # 确保UI响应
        try:
            QApplication.processEvents()
        except Exception as process_e:
            logger.warning(f"处理事件队列失败: {process_e}")




    def clear_all_models(self):
        """清除所有模型"""
        reply = QMessageBox.question(
            self,
            "确认清除",
            "确定要清除所有模型和平面吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            print("[CRITICAL DEBUG] 开始清除所有模型和数据...")
            

            
            # 安全清理法线箭头
            try:
                if hasattr(self, '_normal_arrow_actor'):
                    if hasattr(self.viewer, 'plotter') and hasattr(self.viewer.plotter, 'remove_actor'):
                        try:
                            self.viewer.plotter.remove_actor(self._normal_arrow_actor)
                            self._normal_arrow_actor = None
                            print("[CRITICAL DEBUG] 法线箭头清理完成")
                        except Exception as e:
                            print(f"[CRITICAL WARNING] 移除法线箭头失败: {e}")
            except Exception as normal_e:
                print(f"[CRITICAL WARNING] 清理法线箭头时出错: {normal_e}")
            
            # 清除查看器
            self.viewer.clear_all()
            
            # 重置数据
            self.init_data()
            
            # 重置UI
            self.maxilla_path.setText("未加载")
            self.mandible_path.setText("未加载")
            self.occlusion_path.setText("未加载")
            
            self.progress_bar.setValue(0)
            self.status_label.setText("已清除所有数据")
            
            # 更新按钮状态
            self.update_button_states()
            print("[CRITICAL DEBUG] 清除操作完成")
    
    def reset_view(self):
        """重置视图"""
        try:
            self.viewer.reset_view()
            logger.info("视图已重置")
        except AttributeError:
            logger.warning("查看器不支持reset_view方法")
    
    def _add_view_controls(self, layout):
        """添加视图控制组件"""
        # 创建视图控制组
        view_group = QGroupBox("视图控制")
        view_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        view_group_layout = QVBoxLayout(view_group)
        
        # 视角选择下拉框
        view_layout = QHBoxLayout()
        view_label = QLabel("预设视角:")
        self.view_type_combo = QComboBox()
        # 统一视角选项格式
        self.view_type_combo.addItems([
            "俯视图 (XY)", "前视图 (XZ)", "侧视图 (YZ)", 
            "前", "后", "左", "右", "顶", "底", "等轴测 (ISO)"
        ])
        self.view_type_combo.currentTextChanged.connect(self.on_view_type_changed)
        view_layout.addWidget(view_label)
        view_layout.addWidget(self.view_type_combo)
        
        # 网格显示切换
        self.toggle_grid_btn = QPushButton("关闭网格")
        self.toggle_grid_btn.clicked.connect(self.on_toggle_grid)
        self.toggle_grid_btn.setStyleSheet("padding: 5px;")
        
        # 将组件添加到布局
        view_group_layout.addLayout(view_layout)
        view_group_layout.addWidget(self.toggle_grid_btn)
        
        # 将视图控制组添加到左侧面板
        layout.addWidget(view_group)
    
    def show_screenshot_options(self):
        """显示截图选项对话框，允许用户选择普通截图或高分辨率截图"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QRadioButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("导出截图")
        dialog_layout = QVBoxLayout()
        
        # 截图类型选择
        type_label = QLabel("选择截图类型:")
        dialog_layout.addWidget(type_label)
        
        # 单选按钮组
        type_layout = QVBoxLayout()
        self.normal_screenshot_radio = QRadioButton("普通截图")
        self.normal_screenshot_radio.setChecked(True)
        self.high_res_screenshot_radio = QRadioButton("高分辨率截图")
        type_layout.addWidget(self.normal_screenshot_radio)
        type_layout.addWidget(self.high_res_screenshot_radio)
        dialog_layout.addLayout(type_layout)
        
        # 高分辨率设置
        scale_label = QLabel("分辨率缩放比例:")
        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(1, 10)
        self.scale_spinbox.setValue(2)
        self.scale_spinbox.setSuffix("x")
        self.scale_spinbox.setEnabled(False)  # 默认禁用
        
        # 连接信号，当选择高分辨率时启用缩放比例设置
        self.normal_screenshot_radio.toggled.connect(
            lambda checked: self.scale_spinbox.setEnabled(not checked)
        )
        self.high_res_screenshot_radio.toggled.connect(
            lambda checked: self.scale_spinbox.setEnabled(checked)
        )
        
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_spinbox)
        dialog_layout.addLayout(scale_layout)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        
        dialog.setLayout(dialog_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出截图",
                os.path.join(os.path.expanduser("~"), "screenshot.png"),
                "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*)"
            )
            
            if file_path:
                try:
                    if self.normal_screenshot_radio.isChecked():
                        # 普通截图
                        success = self.viewer.save_screenshot(file_path)
                        title = "普通截图"
                    else:
                        # 高分辨率截图
                        scale = self.scale_spinbox.value()
                        success = self.viewer.capture_high_res_screenshot(file_path, scale)
                        title = f"高分辨率截图 ({scale}x)"
                    
                    if success:
                        QMessageBox.information(self, "导出成功", f"{title}已保存到:\n{file_path}")
                    else:
                        QMessageBox.warning(self, "导出失败", "无法保存截图")
                except Exception as e:
                    # 捕获所有异常，包括方法不存在的情况
                    QMessageBox.warning(self, "导出错误", f"保存截图时发生错误:\n{str(e)}")
                    # 尝试降级到普通截图
                    try:
                        if self.viewer.save_screenshot(file_path):
                            QMessageBox.information(self, "备用成功", f"标准截图已保存到:\n{file_path}\n注意: 高分辨率模式不可用")
                    except:
                        pass
    
    def on_view_type_changed(self, view_text):
        """处理视角类型变化"""
        # 从显示文本映射到实际view_type参数
        view_map = {
            "俯视图 (XY)": "xy",
            "前视图 (XZ)": "xz",
            "侧视图 (YZ)": "yz",
            "前": "front",
            "后": "back",
            "左": "left",
            "右": "right",
            "顶": "top",
            "底": "bottom",
            "等轴测 (ISO)": "iso"
        }
        
        view_type = view_map.get(view_text)
        if view_type:
            try:
                self.viewer.set_view(view_type)
                logger.info(f"切换到视角: {view_text} ({view_type})")
            except AttributeError:
                # 如果查看器不支持set_view方法，忽略错误
                logger.warning("查看器不支持set_view方法")
                pass
    
    def on_toggle_grid(self):
        """处理网格显示切换"""
        current_text = self.toggle_grid_btn.text()
        try:
            if current_text == "关闭网格":
                self.viewer.toggle_grid(False)
                self.toggle_grid_btn.setText("显示网格")
                logger.info("网格已隐藏")
            else:
                self.viewer.toggle_grid(True)
                self.toggle_grid_btn.setText("关闭网格")
                logger.info("网格已显示")
        except AttributeError:
            # 如果查看器不支持toggle_grid方法，忽略错误
            logger.warning("查看器不支持toggle_grid方法")
            pass
    
    # 注：缩放功能已通过鼠标滚轮实现，不再需要单独的按钮和方法
    
    def update_button_states(self):
        """更新按钮状态"""
        # 当有上颌模型且已标记足够点时启用投影按钮
        has_maxilla = "maxilla" in self.viewer.get_models()
        has_enough_points = len(self.viewer.marked_points) >= 3
        has_plane = self.viewer.plane_params is not None
        self.generate_projection_btn.setEnabled(has_maxilla and has_enough_points and has_plane)
        self.produce_projection_btn.setEnabled(has_maxilla and has_enough_points and has_plane)
    
    def update_status(self):
        """更新状态信息"""
        # 可以添加定期检查的代码
        pass
    
    def get_model_type_name(self, model_type):
        """获取模型类型的中文名称"""
        names = {
            "maxilla": "上颌",
            "mandible": "下颌",
            "occlusion": "咬合关系"
        }
        return names.get(model_type, model_type)
    
    def enable_marking(self):
        """启用标记平面功能"""
        # 不再清除已有标记点，保持持久化
        # 移除对clear_marked_points()的调用，确保标记点和平面数据不丢失
        
        # 启用模型查看器的标记功能
        self.viewer.enable_marking("plane")
        
        # 连接标记点更新信号以更新投影按钮状态
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("标记平面功能已启用，右键点击模型表面标记3个点")
        
        print("[MAIN_WINDOW] 标记平面功能已启用 - 标记点和平面将保持持久化")
    
    def enable_maxilla_marking(self):
        """启用标记牙颌功能"""
        # 不再清除已有标记点，保持持久化
        # 移除对clear_marked_points()的调用，确保标记点和平面数据不丢失
        
        # 启用模型查看器的标记功能，指定模式为maxilla
        self.viewer.enable_marking("maxilla")
        
        # 连接标记点更新信号以更新投影按钮状态
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("标记牙颌功能已启用，右键点击牙颌模型表面标记点")
        
        print("[MAIN_WINDOW] 标记牙颌功能已启用 - 标记点和平面将保持持久化")
    
        # 移除了open3d_depth_capture方法
    
    def enable_mandible_crest_marking(self):
        """启用下颌后槽牙槽嵴标记功能"""
        # 不再清除已有标记点，保持持久化
        # 启用模型查看器的标记功能，指定模式为mandible_crest
        self.viewer.enable_marking("mandible_crest")
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("下颌后槽牙槽嵴标记功能已启用，请右键点击下颌模型表面标记左右两侧的后槽牙槽嵴位置")
        
        print("[MAIN_WINDOW] 下颌后槽牙槽嵴标记功能已启用 - 标记点和平面将保持持久化")
    
    def show_mandible_crest_line(self):
        """显示下颌后槽牙槽嵴连线"""
        self.viewer.show_mandible_crest_line()
        self.status_label.setText("已显示下颌后槽牙槽嵴连线")
        self.show_crest_line_btn.setEnabled(True)
        self.project_crest_btn.setEnabled(True)
    
    def project_mandible_crest_to_plane(self):
        """投影后槽牙槽嵴到平面"""
        self.viewer.project_mandible_crest_to_plane()
        self.status_label.setText("已将下颌后槽牙槽嵴连线投影到牙合平面")
    
    def produce_projection_image(self):
        """生成上颌投影图像 - 合并所有功能到一个流程"""
        print("[MAIN_WINDOW] 开始生成上颌投影图像...")
        
        # 设置标记模式为划分上颌模式，确保生成投影时使用上颌模型
        if hasattr(self.viewer, '_marking_mode'):
            self.viewer._marking_mode = "divide_maxilla"
        
        # 1. 显示下颌后槽牙槽嵴连线
        self.viewer.show_mandible_crest_line()
        self.status_label.setText("已显示下颌后槽牙槽嵴连线")
        
        # 2. 投影后槽牙槽嵴到平面
        self.viewer.project_mandible_crest_to_plane()
        self.status_label.setText("已将下颌后槽牙槽嵴连线投影到牙合平面")
        
        # 3. 显示划分上颌连线
        self.viewer.show_divide_maxilla_line()
        self.status_label.setText("已显示划分上颌连线")
        
        # 4. 投影划分上颌到平面
        self.viewer.project_divide_maxilla_to_plane()
        self.status_label.setText("已将划分上颌连线投影到牙合平面")
        
        # 5. 显示上颌牙槽嵴连线
        self.viewer.show_alveolar_ridge_line()
        self.status_label.setText("已显示上颌牙槽嵴连线")
        
        # 6. 投影上颌牙槽嵴到平面
        self.viewer.project_alveolar_ridge_to_plane()
        self.status_label.setText("已将上颌牙槽嵴连线投影到牙合平面")
        
        # 7. 生成投影图像
        self.generate_projection()
        
    def generate_mandible_projection_image(self):
        """生成下颌投影图像 - 合并所有功能到一个流程"""
        print("[MAIN_WINDOW] 开始生成下颌投影图像...")
        
        # 设置标记模式，优先使用已标记的模式
        if hasattr(self.viewer, '_marking_mode'):
            # 检查是否有下颌牙槽嵴标记点，如果有则使用mandible_crest模式
            if hasattr(self.viewer, 'mandible_crest_points') and len(self.viewer.mandible_crest_points) >= 2:
                self.viewer._marking_mode = "mandible_crest"
            else:
                self.viewer._marking_mode = "divide_mandible"
        
        # 1. 显示下颌划分线
        self.viewer.show_divide_mandible_line()
        self.status_label.setText("已显示下颌划分线")
        
        # 2. 投影下颌划分线到牙合平面
        self.viewer.project_divide_mandible_to_plane()
        self.status_label.setText("已将下颌划分线投影到牙合平面")
        
        # 3. 显示牙槽嵴连线（如果有标记）
        self.viewer.show_alveolar_ridge_line()
        self.status_label.setText("已显示牙槽嵴连线")
        
        # 4. 投影牙槽嵴到牙合平面
        self.viewer.project_alveolar_ridge_to_plane()
        self.status_label.setText("已将牙槽嵴连线投影到牙合平面")
        
        # 5. 生成投影图像
        self.generate_projection()
    
    def enable_maxilla_alveolar_ridge_marking(self):
        """启用上颌牙槽嵴多点标记功能"""
        # 不再清除已有标记点，保持持久化
        # 启用模型查看器的标记功能，指定模式为alveolar_ridge（上颌）
        self.viewer.enable_marking("alveolar_ridge")
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("上颌牙槽嵴标记功能已启用，请右键点击模型表面标记多个点")
        
        print("[MAIN_WINDOW] 上颌牙槽嵴标记功能已启用 - 标记点和平面将保持持久化")
    
    def enable_mandible_alveolar_ridge_marking(self):
        """启用下颌牙槽嵴多点标记功能"""
        # 不再清除已有标记点，保持持久化
        # 启用模型查看器的标记功能，指定模式为alveolar_ridge（下颌）
        self.viewer.enable_marking("alveolar_ridge")
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("下颌牙槽嵴标记功能已启用，请右键点击模型表面标记多个点")
        
        print("[MAIN_WINDOW] 下颌牙槽嵴标记功能已启用 - 标记点和平面将保持持久化")
    

    
    def enable_divide_maxilla_marking(self):
        """启用划分上颌多点标记功能"""
        # 不再清除已有标记点，保持持久化
        # 启用模型查看器的标记功能，指定模式为divide_maxilla
        self.viewer.enable_marking("divide_maxilla")
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("划分上颌标记功能已启用，请右键点击模型表面标记点，标记点之间将用直线连接")
        
        print("[MAIN_WINDOW] 划分上颌功能已启用 - 标记点和平面将保持持久化")
        
    def enable_divide_mandible_marking(self):
        """启用划分下颌多点标记功能"""
        # 不再清除已有标记点，保持持久化
        # 启用模型查看器的标记功能，指定模式为divide_mandible
        self.viewer.enable_marking("divide_mandible")
        
        # 连接标记点更新信号
        self.viewer.marked_points_updated.connect(self.update_projection_button_state)
        
        # 更新状态提示
        self.status_label.setText("划分下颌标记功能已启用，请右键点击模型表面标记点，标记点之间将用直线连接")
        
        print("[MAIN_WINDOW] 划分下颌功能已启用 - 标记点和平面将保持持久化")
    
    def generate_projection(self):
        """生成投影图像 - 使用正交投影，在后台线程执行"""
        print("[MAIN_WINDOW] 开始生成正交投影图像...")
        
        # 更新状态和进度条
        self.status_label.setText("正在准备生成正交投影...")
        self.progress_bar.setValue(10)
        
        # 检查模型是否存在，根据当前标记模式检查正确的模型
        required_model = "maxilla"  # 默认检查上颌模型
        if hasattr(self.viewer, '_marking_mode'):
            marking_mode = self.viewer._marking_mode
            if marking_mode in ["divide_mandible", "mandible_crest"]:
                required_model = "mandible"
        
        if required_model not in self.viewer.get_models():
            self._handle_error(f"请先加载{required_model}模型", error_type="warning", log_level="warning")
            return
        
        # 检查是否有拟合平面
        if not self.viewer.plane_params:
            self._handle_error("请先拟合平面", error_type="warning", log_level="warning")
            return
        
        # 检查标记点数量，考虑所有标记模式
        has_valid_points = True  # 根据用户需求：平面拟合完成后即可运行投影功能，无需等待标记点
        
        # 以下代码仅作为参考，不再作为强制检查条件
        if hasattr(self.viewer, '_marking_mode'):
            marking_mode = self.viewer._marking_mode
            
            # 仅记录标记点数量信息，不作为强制条件
            if marking_mode == "maxilla" or marking_mode == "plane":
                # plane或maxilla模式下检查基础标记点
                if len(self.viewer.marked_points) < 3:
                    logger.info("基础标记点不足3个，但仍允许生成投影")
            elif marking_mode == "mandible_crest":
                # 下颌后槽牙槽嵴模式，需要至少2个点
                if hasattr(self.viewer, 'mandible_crest_points') and len(self.viewer.mandible_crest_points) < 2:
                    logger.info("下颌后槽牙槽嵴点不足2个，但仍允许生成投影")
            elif marking_mode == "alveolar_ridge":
                # 牙槽嵴模式，需要至少2个点
                if hasattr(self.viewer, 'alveolar_ridge_points') and len(self.viewer.alveolar_ridge_points) < 2:
                    logger.info("牙槽嵴点不足2个，但仍允许生成投影")
            elif marking_mode == "incisive_papilla":
                # 切牙乳突模式，需要至少1个点
                if len(self.viewer.marked_points) < 1:
                    logger.info("切牙乳突点不足1个，但仍允许生成投影")
            elif marking_mode == "divide_maxilla":
                # 划分上颌模式，需要至少2个点
                if hasattr(self.viewer, 'divide_maxilla_points') and len(self.viewer.divide_maxilla_points) < 2:
                    logger.info("划分上颌点不足2个，但仍允许生成投影")
            elif marking_mode == "divide_mandible":
                # 划分下颌模式，需要至少2个点
                if hasattr(self.viewer, 'divide_mandible_points') and len(self.viewer.divide_mandible_points) < 2:
                    logger.info("划分下颌点不足2个，但仍允许生成投影")
        else:
            # 默认检查基础标记点
            if len(self.viewer.marked_points) < 3:
                logger.info("基础标记点不足3个，但仍允许生成投影")
        
        # 创建后台线程执行投影计算
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class ProjectionThread(QThread):
            """后台执行投影计算的线程"""
            finished = pyqtSignal()
            error = pyqtSignal(str)
            
            def __init__(self, depth_generator):
                super().__init__()
                self.depth_generator = depth_generator
            
            def run(self):
                """执行投影计算"""
                try:
                    # 执行投影计算
                    self.depth_generator.generate_projection(
                        grid_resolution=0.1,  # 添加网格分辨率参数
                        enable_optimization=True,
                        optimization_level='自动',
                        projection_type='orthographic'  # 使用正交投影
                    )
                    self.finished.emit()
                except Exception as e:
                    self.error.emit(str(e))
        
        # 禁用生成按钮，防止重复点击
        self.generate_projection_btn.setEnabled(False)
        self.produce_projection_btn.setEnabled(False)
        self.generate_mandible_projection_btn.setEnabled(False)
        
        # 创建并启动线程
        self.projection_thread = ProjectionThread(self.depth_generator)
        self.projection_thread.finished.connect(self.on_projection_finished)
        self.projection_thread.error.connect(self.on_projection_error)
        self.projection_thread.start()
    
    def on_projection_finished(self):
        """投影计算完成回调"""
        print("[MAIN_WINDOW] 投影计算完成")
        self.status_label.setText("正交投影生成完成")
        self.progress_bar.setValue(100)
        # 重新启用生成按钮
        self.generate_projection_btn.setEnabled(True)
        self.produce_projection_btn.setEnabled(True)
        self.generate_mandible_projection_btn.setEnabled(True)
    
    def on_projection_error(self, error_msg):
        """投影计算错误回调"""
        print(f"[MAIN_WINDOW] 投影计算错误: {error_msg}")
        self._handle_error(f"生成投影时出错: {error_msg}")
        self.progress_bar.setValue(0)
        # 重新启用生成按钮
        self.generate_projection_btn.setEnabled(True)
        self.produce_projection_btn.setEnabled(True)
        self.generate_mandible_projection_btn.setEnabled(True)
        
        print("[MAIN_WINDOW] 投影生成失败")
        return
        
        try:
            self.status_label.setText("正在捕获深度图像...")
            self.progress_bar.setValue(20)
            
            # 使用Open3D手动深度捕获方法获取深度图
            print("[MAIN_WINDOW] 使用Open3D手动深度捕获方法...")
            logger.info("开始Open3D深度图捕获流程")
            
            # 添加详细的计时和状态跟踪
            import time
            start_time = time.time()
            
            # 记录当前环境状态
            logger.debug(f"当前场景状态 - 标记点数量: {len(self.viewer.marked_points)}")
            logger.debug(f"当前场景状态 - 模型加载状态: {bool(self.viewer.get_original_model('maxilla'))}")
            
            # 尝试Open3D深度捕获，如果失败则使用PyVista作为备选
            try:
                result = self.depth_generator.generate_open3d_depth_image()
            except Exception as e:
                logger.warning(f"Open3D深度捕获失败，尝试使用PyVista备选方法: {str(e)}")
                result = self.depth_generator.generate_depth_image_with_pyvista()
            
            elapsed_time = time.time() - start_time
            logger.info(f"深度图捕获耗时: {elapsed_time:.2f}秒")
            
            if result is not None:
                color_img, depth_img, camera_params = result
                
                # 验证结果数据完整性
                logger.debug(f"深度图捕获结果 - 彩色图像形状: {color_img.shape if hasattr(color_img, 'shape') else '未知'}")
                logger.debug(f"深度图捕获结果 - 深度图像形状: {depth_img.shape if hasattr(depth_img, 'shape') else '未知'}")
                logger.debug(f"深度图捕获结果 - 相机参数: {camera_params}")
                
                # 保存Open3D捕获的深度图和相机参数
                self.projection_data['open3d_color_image'] = color_img
                self.projection_data['open3d_depth_image'] = depth_img
                self.projection_data['camera_params'] = camera_params
                
                print("[MAIN_WINDOW] 深度图捕获完成")
                logger.info("深度图捕获成功完成")
            else:
                logger.warning("深度图捕获返回None结果")
                self._handle_error("深度图捕获失败", error_type="warning", log_level="warning")
                # 即使深度图捕获失败，也继续执行后续步骤
        
        except Exception as e:
            logger.error(f"深度图捕获过程中发生异常: {str(e)}", exc_info=True)
            self._handle_error(f"深度图捕获过程中发生异常: {str(e)}", error_type="warning", log_level="error")
            # 即使深度图捕获异常，也继续执行后续步骤
        
        try:
            # 获取上颌原始模型
            maxilla_mesh = self.viewer.get_original_model("maxilla")
            if maxilla_mesh is None:
                self._handle_error("无法获取上颌原始模型", error_type="warning", log_level="warning")
                return
                
            self.status_label.setText("正在处理上颌模型...")
            self.progress_bar.setValue(30)
            print("[MAIN_WINDOW] 成功获取上颌原始模型")
            
            # 使用默认网格分辨率参数
            grid_resolution = 0.2
            
            # 投影到拟合平面
            projection_result = self.viewer.project_maxilla_to_plane(maxilla_mesh, grid_resolution=grid_resolution)
            if projection_result is None:
                QMessageBox.warning(self, "警告", "无法投影到平面")
                return
                
            projected_points_3d, triangles, depth_values = projection_result
            print(f"[MAIN_WINDOW] 成功投影到平面，投影点数量: {len(projected_points_3d)}")
            
            # 检查投影点数量是否足够
            if len(projected_points_3d) < 10:
                QMessageBox.warning(self, "警告", "投影点数量不足")
                return
            
            # 验证数据一致性
            if len(projected_points_3d) != len(depth_values):
                print(f"[MAIN_WINDOW] 数据不一致：投影点({len(projected_points_3d)})与深度值({len(depth_values)})数量不匹配")
                # 取最小值确保数据一致
                min_len = min(len(projected_points_3d), len(depth_values))
                projected_points_3d = projected_points_3d[:min_len]
                depth_values = depth_values[:min_len]
                print(f"[MAIN_WINDOW] 已调整数据长度为: {min_len}")
            
            # 转换为2D坐标
            points_2d = self.viewer.convert_3d_to_2d(projected_points_3d)
            if points_2d is None:
                QMessageBox.warning(self, "警告", "无法转换为2D坐标")
                return
            print(f"[MAIN_WINDOW] 成功转换为2D坐标，2D点数量: {len(points_2d)}")
            
            # === 新增优化步骤 ===
            # 1. 点云预处理
            print("[MAIN_WINDOW] 开始点云预处理...")
            projected_points_3d, depth_values = self.preprocess_point_cloud(
                projected_points_3d, depth_values
            )
            print(f"[MAIN_WINDOW] 预处理后点数量: {len(projected_points_3d)}")
            print(f"[MAIN_WINDOW] 预处理后深度值数量: {len(depth_values)}")
            
            # 检查预处理后的点数量
            if len(projected_points_3d) < 10:
                QMessageBox.warning(self, "警告", "预处理后点云数量不足")
                return
            
            # 更新2D坐标（预处理后）
            points_2d = self.viewer.convert_3d_to_2d(projected_points_3d)
            if points_2d is None:
                QMessageBox.warning(self, "警告", "预处理后无法转换为2D坐标")
                return
            print(f"[MAIN_WINDOW] 更新后2D点数量: {len(points_2d)}")
            
            # 2. 智能分辨率计算 - 确保分辨率不会太小
            optimal_resolution = self.calculate_optimal_resolution(points_2d)
            print(f"[MAIN_WINDOW] UI设置分辨率: {grid_resolution:.2f}mm, 计算最优分辨率: {optimal_resolution:.2f}mm")
            # 使用计算得到的最优分辨率，但不小于默认值的一半
            grid_resolution = max(optimal_resolution, 0.1)  # 确保分辨率不小于0.1mm
            print(f"[MAIN_WINDOW] 最终使用分辨率: {grid_resolution:.2f}mm")
            
            # === 标记线投影处理 ===
            # 获取所有标记线数据并投影到2D平面
            print("[MAIN_WINDOW] 开始标记线投影处理...")
            marker_lines_2d = []
            
            # 处理下颌后槽牙槽嵴连线
            if hasattr(self.viewer, 'mandible_crest_line') and self.viewer.mandible_crest_line:
                print("[MAIN_WINDOW] 处理下颌后槽牙槽嵴连线...")
                crest_line_3d = self.viewer.mandible_crest_line
                # 将连线点投影到拟合平面
                for i in range(len(crest_line_3d) - 1):
                    # 投影每个点到拟合平面
                    p1_proj = self.viewer.project_point_to_plane(crest_line_3d[i])
                    p2_proj = self.viewer.project_point_to_plane(crest_line_3d[i+1])
                    # 转换为2D坐标
                    p1_2d = self.viewer.convert_3d_to_2d_point(p1_proj)
                    p2_2d = self.viewer.convert_3d_to_2d_point(p2_proj)
                    if p1_2d is not None and p2_2d is not None:
                        marker_lines_2d.append({
                            'type': 'mandible_crest',
                            'points': [p1_2d, p2_2d],
                            'color': (1, 0, 1)  # 紫色
                        })
            
            # 处理牙槽嵴拟合曲线
            if hasattr(self.viewer, 'alveolar_ridge_curve') and self.viewer.alveolar_ridge_curve:
                print("[MAIN_WINDOW] 处理牙槽嵴拟合曲线...")
                ridge_curve_3d = self.viewer.alveolar_ridge_curve
                for i in range(len(ridge_curve_3d) - 1):
                    p1_proj = self.viewer.project_point_to_plane(ridge_curve_3d[i])
                    p2_proj = self.viewer.project_point_to_plane(ridge_curve_3d[i+1])
                    p1_2d = self.viewer.convert_3d_to_2d_point(p1_proj)
                    p2_2d = self.viewer.convert_3d_to_2d_point(p2_proj)
                    if p1_2d is not None and p2_2d is not None:
                        marker_lines_2d.append({
                            'type': 'alveolar_ridge',
                            'points': [p1_2d, p2_2d],
                            'color': (0, 1, 0)  # 绿色
                        })
            
            # 处理其他标记线（如果有）
            if hasattr(self.viewer, 'projected_marker_lines') and self.viewer.projected_marker_lines:
                print("[MAIN_WINDOW] 处理其他投影标记线...")
                for line in self.viewer.projected_marker_lines:
                    if 'points' in line and len(line['points']) >= 2:
                        for i in range(len(line['points']) - 1):
                            p1_2d = self.viewer.convert_3d_to_2d_point(line['points'][i])
                            p2_2d = self.viewer.convert_3d_to_2d_point(line['points'][i+1])
                            if p1_2d is not None and p2_2d is not None:
                                marker_lines_2d.append({
                                    'type': line.get('type', 'unknown'),
                                    'points': [p1_2d, p2_2d],
                                    'color': line.get('color', (1, 0, 0))  # 默认红色
                                })
            
            print(f"[MAIN_WINDOW] 成功投影 {len(marker_lines_2d)} 条标记线")
            
            # 保存映射关系和标记线数据
            self.projection_data['points_2d'] = points_2d
            self.projection_data['projected_points_3d'] = projected_points_3d
            self.projection_data['depth_values'] = depth_values
            self.projection_data['marker_lines_2d'] = marker_lines_2d
            
            # 使用默认参数
            interpolation_method = "linear"
            colormap = 'gray'  # 强制使用灰度图
            optimization_level = "自动"
            
            # 使用优化的深度图生成方法，结合Open3D捕获的深度图
            print("[MAIN_WINDOW] 使用优化的深度图生成方法，结合Open3D捕获的深度图")
            print(f"[MAIN_WINDOW] 优化级别: {optimization_level}")
            
            # 显示进度信息
            self.statusBar().showMessage("正在生成投影图像，请稍候...")
            
            # 创建处理线程
            def generate_depth_image():
                """在后台线程中生成深度图像"""
                try:
                    # 生成优化的深度图，结合Open3D捕获的深度数据
                    result, error = self.generate_optimized_depth_image(
                        points_2d, depth_values, grid_resolution, 
                        interpolation_method, colormap, optimization_level
                    )
                    if result:
                        # 将结果保存到projection_data中
                        self.projection_data['depth_image'] = result['depth_image']
                        self.projection_data['extent'] = result['extent']
                        self.projection_data['depth_min'] = result['depth_min']
                        self.projection_data['depth_max'] = result['depth_max']
                        return result
                    else:
                        return error
                except Exception as e:
                    print(f"[MAIN_WINDOW] 生成深度图失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return str(e)
            
            # 定义线程完成后的处理函数
            def on_generation_finished(result):
                """处理线程完成事件"""
                # 更新状态条
                self.statusBar().showMessage("")
                
                print(f"[MAIN_WINDOW] 线程完成，结果类型: {type(result)}, 内容: {result.keys() if isinstance(result, dict) else result}")
                
                if isinstance(result, dict):
                    # 检查是否生成了深度图
                    depth_image = result.get('depth_image')
                    extent = result.get('extent')
                    marker_lines_2d = result.get('marker_lines_2d')
                    
                    print(f"[MAIN_WINDOW] 深度图数据: {depth_image is not None}, 形状: {depth_image.shape if depth_image is not None else 'None'}")
                    print(f"[MAIN_WINDOW] 范围: {extent}, 标记线: {marker_lines_2d}")
                    
                    # 添加调试信息，检查标记线坐标范围
                    if marker_lines_2d is not None and len(marker_lines_2d) > 0:
                        all_points = []
                        for line in marker_lines_2d:
                            if 'points' in line:
                                all_points.extend(line['points'])
                        all_points = np.array(all_points)
                        print(f"[MAIN_WINDOW] 标记线坐标范围: x [{all_points[:,0].min():.2f}, {all_points[:,0].max():.2f}], y [{all_points[:,1].min():.2f}, {all_points[:,1].max():.2f}]")
                        print(f"[MAIN_WINDOW] 深度图范围: {extent}")
                    
                    if depth_image is not None and extent is not None:
                        # 显示增强的深度图对话框
                        try:
                            self.depth_dialog = EnhancedDepthImageDialog(
                                depth_image=depth_image,
                                extent=extent,
                                marker_lines_2d=marker_lines_2d,
                                parent=self
                            )
                            self.depth_dialog.setWindowTitle("投影深度图 - 二维平面")
                            self.depth_dialog.setModal(False)
                            self.depth_dialog.show()
                            print("[MAIN_WINDOW] 深度图对话框显示成功")
                        except Exception as dialog_e:
                            print(f"[MAIN_WINDOW] 创建/显示深度图对话框失败: {dialog_e}")
                            import traceback
                            traceback.print_exc()
                            QMessageBox.warning(self, "警告", f"显示深度图失败: {str(dialog_e)}")
                    else:
                        QMessageBox.warning(self, "警告", "未生成有效的深度图数据")
                else:
                    # 显示错误信息
                    QMessageBox.warning(self, "警告", f"生成投影图像失败: {result}")
            
            # 启动后台线程
            self.processing_thread = ProcessingThread(generate_depth_image)
            self.processing_thread.processing_finished.connect(on_generation_finished)
            self.processing_thread.error_occurred.connect(lambda error: QMessageBox.warning(self, "警告", f"生成投影图像失败: {error}"))
            self.processing_thread.start()
            print("[MAIN_WINDOW] 后台线程启动，开始生成深度图")
            
            # 只隐藏主要模型（上颌、下颌、咬合），确保用户关注二维投影图像
            # 保存当前模型的可见性状态
            self.saved_model_visibility = {}
            for model_name in list(self.viewer.model_actors.keys()):
                # 只隐藏主要模型，保留辅助模型的状态
                if model_name in ["maxilla", "mandible", "occlusion"]:
                    self.saved_model_visibility[model_name] = True  # 这些模型将被隐藏
                    self.viewer.toggle_model_visibility(model_name, False)
            print(f"[MAIN_WINDOW] 隐藏了主要三维模型，保留辅助模型")
            
        except Exception as e:
            print(f"[MAIN_WINDOW] 生成投影图像失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"生成投影图像时发生错误: {str(e)}")
    
    def _downsample_point_cloud(self, points, target_points=100000):
        """对点云进行降采样以减少点数量，提高性能和减少内存占用
        
        Args:
            points: 点云数组
            target_points: 目标点数量
            
        Returns:
            降采样后的点云数组
        """
        if len(points) <= target_points:
            return points
            
        # 使用随机采样方法进行降采样
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]
    
    def _uniform_sample_points(self, points, target_points=100000):
        """均匀采样点集，确保采样密度和覆盖度
        
        Args:
            points: 点云数组
            target_points: 目标点数量
            
        Returns:
            采样后的点云数组
        """
        if len(points) <= target_points:
            return points
            
        # 使用随机采样方法进行均匀采样
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]
    
    def preprocess_point_cloud(self, points_3d, depth_values): 
        """增强的点云预处理"""
        # 保存原始数据作为后备
        original_points = points_3d
        original_depth = depth_values
        
        try:
            import open3d as o3d 
            
            # 转换为Open3D点云 
            pcd = o3d.geometry.PointCloud() 
            pcd.points = o3d.utility.Vector3dVector(points_3d) 
            
            # 统计离群点移除 
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) 
            filtered_pcd = pcd.select_by_index(ind) 
            filtered_points = np.asarray(filtered_pcd.points)
            
            # 检查过滤后是否有点
            if len(filtered_points) == 0:
                print("[WARNING] 离群点过滤后点云为空，返回原始数据")
                return original_points, original_depth
            
            # 体素下采样保持均匀密度 
            downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.1) 
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # 检查下采样后是否有点
            if len(downsampled_points) == 0:
                print("[WARNING] 体素下采样后点云为空，返回过滤后数据")
                return filtered_points, depth_values[ind]
            
            # 对于下采样后的点，使用最近邻方法匹配深度值
            from scipy.spatial import KDTree
            tree = KDTree(filtered_points)
            _, indices = tree.query(downsampled_points, k=1)
            
            filtered_depth = depth_values[ind]
            downsampled_depth = filtered_depth[indices]
            
            return downsampled_points, downsampled_depth 
            
        except ImportError:
            # 回退到简单的统计过滤 
            print("[WARNING] Open3D不可用，使用简单过滤") 
            from scipy import stats 
            z_scores = np.abs(stats.zscore(depth_values)) 
            valid_mask = z_scores < 3 
            
            filtered_points = points_3d[valid_mask]
            filtered_depth = depth_values[valid_mask]
            
            # 检查过滤后是否有点
            if len(filtered_points) == 0:
                print("[WARNING] 统计过滤后点云为空，返回原始数据")
                return original_points, original_depth
            
            return filtered_points, filtered_depth
    
    def calculate_optimal_resolution(self, points_2d): 
        """根据点云密度自动计算最优网格分辨率""" 
        # 确保points_2d不为空
        if points_2d is None or len(points_2d) == 0:
            print("[WARNING] 点云为空，使用默认分辨率")
            return 0.2
        
        # 计算点云边界 
        x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
        y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 检查边界是否有效
        if x_range < 1e-6 or y_range < 1e-6:
            print("[WARNING] 点云范围过小，使用默认分辨率")
            return 0.2
        
        # 基于点云密度估算分辨率 
        num_points = len(points_2d) 
        area = x_range * y_range 
        point_density = num_points / area if area > 0 else 1 
        
        # 动态调整分辨率 
        if point_density > 100:  # 高密度点云 
            resolution = 0.1 
        elif point_density > 50:  # 中密度点云 
            resolution = 0.2 
        else:  # 低密度点云 
            resolution = 0.3 
        
        # 确保分辨率合理 
        resolution = max(0.05, min(resolution, 1.0)) 
        
        print(f"[OPTIMIZATION] 点云密度: {point_density:.2f} points/mm², 推荐分辨率: {resolution:.2f}mm") 
        return resolution
    
    def advanced_interpolation(self, points_2d, depth_values, xi, yi): 
        """增强的插值算法 - 优化版""" 
        from scipy.interpolate import griddata, Rbf 
        import warnings 
        warnings.filterwarnings('ignore') 
        
        # 先对输入数据进行统计分析
        mean_depth = np.mean(depth_values)
        std_depth = np.std(depth_values)
        
        # 第一遍：三次插值 - 提供更平滑的结果
        zi = griddata(points_2d, depth_values, (xi, yi), method='cubic') 
        
        # 处理NaN值 - 分层策略
        nan_mask = np.isnan(zi) 
        if np.any(nan_mask): 
            # 先尝试使用多二次曲面径向基函数插值处理NaN区域，这对不规则点云效果更好
            try:
                rbf = Rbf(points_2d[:, 0], points_2d[:, 1], depth_values, function='multiquadric', epsilon=2*std_depth)
                zi_rbf = rbf(xi, yi)
                # 只填充NaN区域
                zi[nan_mask] = zi_rbf[nan_mask]
            except:
                try:
                    # 如果多二次曲面失败，尝试高斯径向基函数
                    rbf = Rbf(points_2d[:, 0], points_2d[:, 1], depth_values, function='gaussian', epsilon=std_depth)
                    zi_rbf = rbf(xi, yi)
                    zi[nan_mask] = zi_rbf[nan_mask]
                except:
                    # 最后回退到最近邻
                    zi_nn = griddata(points_2d, depth_values, (xi, yi), method='nearest') 
                    zi[nan_mask] = zi_nn[nan_mask] 
        
        # 边缘保持平滑 - 使用优化后的算法
        zi = self.edge_preserving_smooth(zi) 
        
        # 深度图质量增强（动态范围调整）
        zi = self.enhance_depth_map_quality(zi, depth_values)        
        
        # 细节增强
        zi = self.enhance_depth_details(zi)
        
        return zi 

    def edge_preserving_smooth(self, depth_map): 
        """高级边缘保持平滑 - 优化版"""
        from scipy.ndimage import gaussian_filter, median_filter, sobel
        import cv2
        
        # 先使用中值滤波去除椒盐噪声
        median_filtered = median_filter(depth_map, size=3)
        
        # 计算深度图的梯度（边缘）
        depth_grad_x = sobel(median_filtered, axis=0)
        depth_grad_y = sobel(median_filtered, axis=1)
        depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        depth_grad_mag = (depth_grad_mag - np.min(depth_grad_mag)) / (np.max(depth_grad_mag) - np.min(depth_grad_mag)) if np.max(depth_grad_mag) > np.min(depth_grad_mag) else depth_grad_mag
        
        # 转换为8位图像以便使用OpenCV的双边滤波
        depth_min = np.min(median_filtered)
        depth_max = np.max(median_filtered)
        if depth_max > depth_min:
            depth_8bit = ((median_filtered - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_8bit = median_filtered.astype(np.uint8)
        
        # 自适应双边滤波参数
        # 根据深度梯度调整滤波强度
        edge_strength = np.max(depth_grad_mag)
        d = 7 if edge_strength > 0.5 else 5  # 边缘越强，滤波窗口越小
        sigmaColor = 20 if edge_strength > 0.5 else 25  # 边缘越强，颜色相似度影响越大
        sigmaSpace = 15 if edge_strength > 0.5 else 20  # 边缘越强，空间距离影响越小
        
        # 应用双边滤波 - 保持边缘同时平滑区域
        try:
            smoothed_8bit = cv2.bilateralFilter(depth_8bit, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
            # 转换回原始范围
            smoothed = (smoothed_8bit / 255.0) * (depth_max - depth_min) + depth_min
            
            # 边缘增强 - 根据梯度信息添加边缘强调
            edge_enhancement = 0.15 * (depth_map - smoothed)
            result = smoothed + edge_enhancement
            result = np.clip(result, depth_min, depth_max)
        except:
            # 如果OpenCV不可用或失败，使用改进的混合滤波
            # 应用两个不同参数的高斯滤波
            gaussian_smooth = gaussian_filter(median_filtered, sigma=0.8)
            gaussian_sharp = gaussian_filter(median_filtered, sigma=0.3)
            
            # 根据梯度混合结果
            alpha = np.clip(depth_grad_mag * 2, 0, 1)  # 边缘区域保留更多细节
            result = alpha * gaussian_sharp + (1 - alpha) * gaussian_smooth
            
            # 最终混合原始图像以保持最大细节
            result = 0.3 * median_filtered + 0.7 * result
        
        return result
    
    def enhance_depth_map_quality(self, zi, depth_values): 
        """深度图质量增强 - 自适应对比度增强""" 
        import cv2
        
        # 1. 动态范围调整 - 使用更精确的百分位
        # 对于深度图，使用1%和99%百分位可以更好地保留细节
        depth_1 = np.percentile(depth_values, 1)
        depth_99 = np.percentile(depth_values, 99)
        
        # 2. 压缩极端值，增强中间范围对比度
        zi_clipped = np.clip(zi, depth_1, depth_99)
        
        # 3. 线性拉伸到完整范围
        if depth_99 > depth_1:
            zi_stretched = (zi_clipped - depth_1) / (depth_99 - depth_1)
            
            # 4. 应用CLAHE（对比度受限的自适应直方图均衡化）
            try:
                # 转换为8位图像以便使用OpenCV的CLAHE
                zi_8bit = (zi_stretched * 255).astype(np.uint8)
                
                # 创建CLAHE对象
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                zi_clahe = clahe.apply(zi_8bit)
                
                # 转换回浮点数
                zi_enhanced = zi_clahe / 255.0
                
                # 5. 应用轻微的非线性增强以提升视觉效果
                # 使用伽马校正，根据深度分布调整伽马值
                mean_depth = np.mean(zi_stretched)
                gamma = 1.1 if mean_depth < 0.5 else 0.9
                zi_enhanced = np.power(zi_enhanced, gamma)
            except:
                # 如果CLAHE不可用，使用改进的非线性增强
                # 基于直方图均衡化的近似实现
                hist, bins = np.histogram(zi_stretched, bins=256, range=(0, 1))
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                
                # 使用累积分布函数进行直方图均衡化
                zi_enhanced = np.interp(zi_stretched.flatten(), bins[:-1], cdf_normalized)
                zi_enhanced = zi_enhanced.reshape(zi_stretched.shape)
                
                # 应用轻微的伽马校正
                zi_enhanced = np.power(zi_enhanced, 0.95)
        else:
            zi_enhanced = zi_clipped
            # 如果范围太小，归一化到0-1区间
            if np.max(zi_enhanced) > np.min(zi_enhanced):
                zi_enhanced = (zi_enhanced - np.min(zi_enhanced)) / (np.max(zi_enhanced) - np.min(zi_enhanced))
        
        return zi_enhanced
        
    def enhance_depth_details(self, depth_map):
        """增强深度图细节 - 优化版"""
        from scipy.ndimage import gaussian_filter, sobel, laplace
        
        # 1. 使用多尺度高斯滤波提取细节
        # 创建不同尺度的模糊图像
        blur_small = gaussian_filter(depth_map, sigma=0.8)  # 小尺度模糊，提取细细节
        blur_medium = gaussian_filter(depth_map, sigma=2.0)  # 中尺度模糊，提取中等细节
        blur_large = gaussian_filter(depth_map, sigma=4.0)  # 大尺度模糊，提取粗细节
        
        # 2. 提取不同尺度的细节
        details_small = depth_map - blur_small  # 细细节
        details_medium = blur_small - blur_medium  # 中等细节
        details_large = blur_medium - blur_large  # 粗细节
        
        # 3. 计算边缘图以指导细节增强
        # 使用拉普拉斯算子检测边缘
        edge_map = np.abs(laplace(depth_map))
        edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map) + 1e-10)
        
        # 4. 自适应细节增强
        # 根据边缘强度和细节尺度调整增强幅度
        alpha_small = 1.8  # 细细节增强系数
        alpha_medium = 1.2  # 中等细节增强系数
        alpha_large = 0.8  # 粗细节增强系数
        
        # 边缘区域增强更多细节，平坦区域增强较少
        enhanced_details_small = details_small * (alpha_small * edge_map + 0.2)
        enhanced_details_medium = details_medium * (alpha_medium * edge_map + 0.3)
        enhanced_details_large = details_large * (alpha_large * edge_map + 0.5)
        
        # 5. 组合所有细节
        total_details = enhanced_details_small + enhanced_details_medium + enhanced_details_large
        
        # 6. 应用细节增益控制
        # 限制细节增强的最大幅度，避免过度增强噪声
        max_detail_gain = 0.15
        total_details = np.clip(total_details, -max_detail_gain, max_detail_gain)
        
        # 7. 重新组合原始图像和增强的细节
        result = depth_map + total_details
        
        # 8. 确保结果在有效范围内
        result = np.clip(result, 0, 1)
        
        # 9. 应用轻微的锐化以增强整体清晰度
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]) / 9.0
        from scipy.signal import convolve2d
        result = convolve2d(result, sharpen_kernel, mode='same', boundary='symm')
        result = np.clip(result, 0, 1)
        
        return result
    
    def clear_marked_points(self):
        """清除所有标记点"""
        self.viewer.clear_marked_points()
        self.status_label.setText("已清除所有标记点")
        
        # 更新投影按钮状态
        self.update_projection_button_state()
    
    def on_fit_plane_completed(self, plane_params):
        """平面拟合完成处理"""
        self.plane_params = plane_params
        self.status_label.setText("平面拟合完成")
        
        print(f"[MAIN_WINDOW] 平面拟合完成，参数: {plane_params}")
        
        # 更新生成投影按钮状态
        self.update_projection_button_state()
    
    def generate_optimized_depth_image(self, points_2d, depth_values, grid_resolution=0.05, interpolation_method='cubic', colormap='gray', optimization_level='自动'): 
         """优化的深度图生成方法，支持标记线绘制""" 
         try: 
             import matplotlib 
             matplotlib.use('Agg') 
             import matplotlib.pyplot as plt 
             from scipy.interpolate import griddata 
              
             # 检查输入数据 
             if points_2d is None or len(points_2d) == 0: 
                 print("[MAIN_WINDOW] 警告: 投影点集为空") 
                 return None, "投影点集为空" 
              
             if depth_values is None or len(depth_values) == 0: 
                 print("[MAIN_WINDOW] 警告: 深度值为空") 
                 return None, "深度值为空" 
              
             # 确保points_2d是2D数组 
             if len(points_2d.shape) != 2 or points_2d.shape[1] != 2: 
                 print("[MAIN_WINDOW] 警告: 点云数据格式错误") 
                 return None, "点云数据格式错误" 
              
             # 数据一致性检查 
             min_len = min(len(points_2d), len(depth_values)) 
             points_2d, depth_values = points_2d[:min_len], depth_values[:min_len] 
              
             # 确保过滤后的点数量足够 
             if len(points_2d) < 10: 
                 print("[MAIN_WINDOW] 警告: 有效点数量不足，无法生成深度图") 
                 return None, "有效点数量不足，无法生成深度图" 
              
             # 根据优化级别调整参数 
             if optimization_level == '高质量':
                 # 高质量设置：更严格的异常点过滤，更多的采样点，更高的插值质量
                 print("[MAIN_WINDOW] 高质量优化模式：更严格的异常点过滤")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 2*z_std) & (depth_values <= z_mean + 2*z_std)  # 更严格的过滤
                 max_points = 100000  # 更多的采样点
                 interpolation_method = 'cubic'  # 更高质量的插值
             elif optimization_level == '平衡':
                 # 平衡设置：默认参数
                 print("[MAIN_WINDOW] 平衡优化模式：默认参数")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
                 max_points = 50000
             elif optimization_level == '快速':
                 # 快速设置：更宽松的异常点过滤，更少的采样点，更快的插值
                 print("[MAIN_WINDOW] 快速优化模式：宽松的异常点过滤，更少的采样点")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 5*z_std) & (depth_values <= z_mean + 5*z_std)  # 更宽松的过滤
                 max_points = 20000  # 更少的采样点
                 interpolation_method = 'linear'  # 更快的插值
             else:  # 自动
                 # 自动设置：根据数据量和分布自动调整
                 print("[MAIN_WINDOW] 自动优化模式：根据数据自动调整参数")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
                 # 根据数据量自动调整采样点数量
                 if len(points_2d) > 100000:
                     max_points = 50000
                 else:
                     max_points = 100000
              
             # 异常点过滤 
             points_2d = points_2d[valid_mask] 
             depth_values = depth_values[valid_mask] 
              
             # 确保过滤后的点数量足够 
             if len(points_2d) < 10: 
                 print("[MAIN_WINDOW] 警告：过滤后点数量不足，跳过降采样") 
                 # 使用所有点，不进行降采样 
             else: 
                 # 性能优化：降采样 
                 if len(points_2d) > max_points: 
                     indices = np.random.choice(len(points_2d), max_points, replace=False) 
                     points_2d = points_2d[indices] 
                     depth_values = depth_values[indices] 
             
             # 计算范围 
             x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0]) 
             y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1]) 
             
             # 初步创建网格
             xi = np.arange(x_min - 2, x_max + 2, grid_resolution) 
             yi = np.arange(y_min - 2, y_max + 2, grid_resolution) 
             
             # 计算理论内存占用（每个浮点数据4字节）
             grid_size = len(xi) * len(yi)
             estimated_memory_mb = (grid_size * 4) / (1024 * 1024)  # MB
             max_memory_mb = 500  # 最大允许内存占用
             
             if estimated_memory_mb > max_memory_mb:
                 print(f"[MAIN_WINDOW] 警告: 预计内存占用 {estimated_memory_mb:.2f} MB 超过阈值 {max_memory_mb} MB")
                 # 自动降低分辨率
                 new_resolution = grid_resolution * 2
                 print(f"[MAIN_WINDOW] 自动将分辨率降低到 {new_resolution:.1f} mm")
                 
                 # 重新生成网格
                 xi = np.arange(x_min - 2, x_max + 2, new_resolution)
                 yi = np.arange(y_min - 2, y_max + 2, new_resolution)
                 grid_resolution = new_resolution
                 
                 # 重新计算内存占用
                 grid_size = len(xi) * len(yi)
                 estimated_memory_mb = (grid_size * 4) / (1024 * 1024)  # MB
                 print(f"[MAIN_WINDOW] 调整后预计内存占用 {estimated_memory_mb:.2f} MB")
             
             # 真正创建网格
             xi, yi = np.meshgrid(xi, yi)
             print(f"[MAIN_WINDOW] 平面离散化完成，网格大小: {xi.shape[0]}x{xi.shape[1]}")
             
             # 使用高级插值 
             zi = self.advanced_interpolation(points_2d, depth_values, xi, yi) 
              
             # 确保插值结果不为空 
             if zi is None or np.isnan(zi).all(): 
                 print("[MAIN_WINDOW] 警告: 深度图插值失败") 
                 return None, "深度图插值失败" 
              
             # 处理插值结果中的NaN值 
             if np.isnan(zi).any(): 
                 print("[MAIN_WINDOW] 警告: 插值结果包含NaN值，将使用最近邻填充") 
                 # 使用最近邻插值填充NaN值 
                 from scipy.interpolate import griddata 
                 zi_nn = griddata(points_2d, depth_values, (xi, yi), method='nearest') 
                 zi[np.isnan(zi)] = zi_nn[np.isnan(zi)] 
              
             # 质量增强 
             zi = self.enhance_depth_map_quality(zi, depth_values)
             
             # 添加深度值统计信息
             depth_stats = {
                 'min': np.min(depth_values),
                 'max': np.max(depth_values),
                 'mean': np.mean(depth_values),
                 'median': np.median(depth_values),
                 'std': np.std(depth_values)
             }
             print(f"[MAIN_WINDOW] 深度值统计: {depth_stats}")
             
             # 计算点云密度
             area = (x_max - x_min) * (y_max - y_min)
             if area > 0:
                 density = len(points_2d) / area
             else:
                 density = 0
             print(f"[MAIN_WINDOW] 点云密度: {density:.2f} points/mm²")
             
             # 获取深度值范围
             depth_min = np.min(depth_values)
             depth_max = np.max(depth_values)
             print(f"[MAIN_WINDOW] 深度值范围: [{depth_min:.2f}, {depth_max:.2f} mm]")
             
             # 创建图形
             fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
             print(f"[MAIN_WINDOW] 创建图形完成，画布大小: 12x10, DPI: 200")
             
             # 绘制真正的深度图
             print(f"[MAIN_WINDOW] 开始绘制深度图，颜色映射: {colormap}...")
              
             # 检查是否有标记线需要绘制
             marker_lines_2d = None
             if hasattr(self, 'projection_data') and 'marker_lines_2d' in self.projection_data:
                 marker_lines_2d = self.projection_data['marker_lines_2d']
                 print(f"[MAIN_WINDOW] 发现标记线数据，准备绘制 {len(marker_lines_2d)} 条线")
              
             # 绘制深度图
             # 使用xi和yi的实际范围作为extent参数，确保坐标正确匹配
             # 由于应用了深度图增强，使用增强后的数据范围
             enhanced_min = np.min(zi)
             enhanced_max = np.max(zi)
             im = ax.imshow(zi, cmap=colormap, origin='upper', 
                          extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                          vmin=enhanced_min, vmax=enhanced_max, aspect='equal',
                          interpolation='lanczos')  # 使用更平滑的lanczos插值，进一步提高图像连续性
             
             # 绘制标记线
             if marker_lines_2d is not None:
                 for line_data in marker_lines_2d:
                     if 'points' in line_data:
                         points = np.array(line_data['points'])
                         color = line_data.get('color', (1, 0, 0))  # 默认红色
                         label = line_data.get('label', '')
                          
                         # 绘制线段
                         ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2, alpha=0.8)
                         print(f"[MAIN_WINDOW] 已绘制标记线: {label}, 颜色: {color}, 点数: {len(points)}")
                          # 如果有标签，添加到线段中间位置
                         if label:
                             mid_idx = len(points) // 2
                             mid_point = points[mid_idx]
                             ax.text(mid_point[0], mid_point[1], label, fontsize=8, 
                                   color=color, backgroundcolor='white', alpha=0.7, 
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
             
             # 添加详细的调试信息
             print(f"[MAIN_WINDOW] 深度图绘制详细信息:")
             print(f"[MAIN_WINDOW]  - 原始深度值范围: [{depth_min:.2f}, {depth_max:.2f}] mm")
             print(f"[MAIN_WINDOW]  - 插值后深度值范围: [{np.nanmin(zi):.2f}, {np.nanmax(zi):.2f}] mm")
             print(f"[MAIN_WINDOW]  - 增强后深度值范围: [{enhanced_min:.4f}, {enhanced_max:.4f}] mm")
             print(f"[MAIN_WINDOW]  - 网格范围(xi): [{xi.min():.2f}, {xi.max():.2f}] mm ({len(xi)}个点)")
             print(f"[MAIN_WINDOW]  - 网格范围(yi): [{yi.min():.2f}, {yi.max():.2f}] mm ({len(yi)}个点)")
             print(f"[MAIN_WINDOW]  - 显示范围(extent): [{xi.min():.2f}, {xi.max():.2f}, {yi.min():.2f}, {yi.max():.2f}]")
             print(f"[MAIN_WINDOW]  - 颜色映射: {colormap}")
             print(f"[MAIN_WINDOW]  - 插值方法: bilinear")
             print(f"[MAIN_WINDOW]  - 坐标原点: lower")
             
             # 保存插值坐标用于联动
             self.projection_data['interpolation_coords'] = (xi, yi)
             self.projection_data['depth_image'] = zi  # 存储numpy数组而不是matplotlib图像对象
             self.projection_data['depth_ax'] = ax
             self.projection_data['depth_min'] = depth_min
             self.projection_data['depth_max'] = depth_max
             self.projection_data['extent'] = [x_min - 2, x_max + 2, y_min - 2, y_max + 2]
             
             # 添加鼠标点击事件处理，实现深度图与3D模型联动
             def on_depth_image_click(event):
                 if event.inaxes != ax:
                     return
                 
                 # 获取点击的2D坐标
                 x_click = event.xdata
                 y_click = event.ydata
                 
                 # 找到离点击位置最近的2D点
                 if self.projection_data['points_2d'] is not None:
                     points_2d = self.projection_data['points_2d']
                     distances = np.sqrt((points_2d[:, 0] - x_click)**2 + (points_2d[:, 1] - y_click)**2)
                     min_idx = np.argmin(distances)
                     
                     # 获取对应的3D点
                     projected_points_3d = self.projection_data['projected_points_3d']
                     if min_idx < len(projected_points_3d):
                         clicked_point_3d = projected_points_3d[min_idx]
                         
                         # 在3D视图中高亮显示这个点
                         self.viewer.highlight_point(clicked_point_3d, color=[1, 0, 0], size=3)
                         print(f"[MAIN_WINDOW] 深度图点击位置({x_click:.2f}, {y_click:.2f})对应3D点: {clicked_point_3d}")
             
             # 连接鼠标点击事件
             fig.canvas.mpl_connect('button_press_event', on_depth_image_click)
             
             # 确保颜色映射符合预期：距离越近越亮，越远越暗
             # gray_r 映射：深度值越大（距离越远），颜色越暗；深度值越小（距离越近），颜色越亮
             
             # 添加颜色条，但不显示标签和刻度
             cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
             cbar.set_label('')
             cbar.set_ticks([])
              
             print("[MAIN_WINDOW] 深度图绘制完成")
              
             # 确保输出目录存在
             output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
             output_dir = os.path.join(output_dir, "output")
             os.makedirs(output_dir, exist_ok=True)
             
             # 生成时间戳，确保文件名唯一
             import datetime
             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式：YYYYMMDD_HHMMSS_mmm
              
             # 保存深度图文件，使用时间戳生成唯一文件名
             output_path_png = os.path.join(output_dir, f"projection_{timestamp}.png")
             output_path_svg = os.path.join(output_dir, f"projection_{timestamp}.svg")
             output_path_depth = os.path.join(output_dir, f"depth_map_{timestamp}.png")
             
             # 保存深度图为高DPI的PNG
             print(f"[MAIN_WINDOW] 准备保存深度图到: {output_path_depth}")
             try:
                 print("[MAIN_WINDOW] 开始保存深度图...")
                 plt.savefig(output_path_depth, dpi=500, bbox_inches='tight', pad_inches=0.1, format='png', 
                           pil_kwargs={'quality': 95, 'optimize': True}, antialiased=True, rasterized=True)
                 print(f"[MAIN_WINDOW] 深度图保存完成: {output_path_depth}")
             except Exception as e:
                 print(f"[MAIN_WINDOW] 保存深度图失败: {e}")
              
             # 保存投影图像为SVG
             print(f"[MAIN_WINDOW] 准备保存投影图像到: {output_path_svg}")
             try:
                 print("[MAIN_WINDOW] 开始保存SVG图像...")
                 plt.savefig(output_path_svg, format='svg', bbox_inches='tight', pad_inches=0.1)
                 print(f"[MAIN_WINDOW] SVG图像保存完成: {output_path_svg}")
             except Exception as e:
                 print(f"[MAIN_WINDOW] 保存SVG图像失败: {e}")
             
             # 关闭图形
             plt.close(fig)
              
             # 保存其他数据文件（与原始方法保持一致）
             try:
                 # 确保projection_data存在
                 if hasattr(self, 'projection_data'):
                     # 保存深度矩阵为npy文件（用于后续分析）
                     output_path_depth_matrix = os.path.join(output_dir, f"depth_matrix_{timestamp}.npy")
                     print(f"[MAIN_WINDOW] 准备保存深度矩阵到: {output_path_depth_matrix}")
                     np.save(output_path_depth_matrix, zi)
                     print(f"[MAIN_WINDOW] 深度矩阵保存完成")
             except Exception as e:
                 print(f"[MAIN_WINDOW] 保存深度矩阵失败: {e}")
             
             # 保存深度值为CSV格式（便于通用软件分析）
             output_path_csv = os.path.join(output_dir, f"depth_values_{timestamp}.csv")
             print(f"[MAIN_WINDOW] 准备保存深度值CSV到: {output_path_csv}")
             try:
                 print("[MAIN_WINDOW] 开始保存深度值CSV...")
                 # 创建包含2D坐标和对应深度值的结构化数据
                 csv_data = np.column_stack((self.projection_data['points_2d'], self.projection_data['depth_values']))
                 header = "x,y,depth"
                 np.savetxt(output_path_csv, csv_data, delimiter=",", header=header, comments="")
                 print(f"[MAIN_WINDOW] 深度值CSV保存完成")
             except Exception as e:
                 print(f"[MAIN_WINDOW] 保存深度值CSV失败: {e}")
             
             # 保存投影点云为PLY格式（便于3D软件查看）
             output_path_ply = os.path.join(output_dir, "projected_points.ply")
             print(f"[MAIN_WINDOW] 准备保存投影点云到: {output_path_ply}")
             try:
                 print("[MAIN_WINDOW] 开始保存投影点云...")
                 # 创建Open3D点云对象
                 pcd = o3d.geometry.PointCloud()
                 pcd.points = o3d.utility.Vector3dVector(self.projection_data['projected_points_3d'])
                 # 将深度值作为颜色信息（标准化）
                 colors = np.zeros_like(self.projection_data['projected_points_3d'])
                 depth_values = self.projection_data['depth_values']
                 norm_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))
                 colors[:, 0] = norm_depth  # 使用红色通道表示深度值
                 pcd.colors = o3d.utility.Vector3dVector(colors)
                 # 保存PLY文件
                 o3d.io.write_point_cloud(output_path_ply, pcd)
                 print(f"[MAIN_WINDOW] 投影点云保存完成")
             except Exception as e:
                 print(f"[MAIN_WINDOW] 保存投影点云失败: {e}")
             
             # 准备返回结果数据 
             result = {
                 'depth_image': zi,
                 'extent': [xi.min(), xi.max(), yi.min(), yi.max()],
                 'depth_min': depth_min,
                 'depth_max': depth_max,
                 'output_path_depth': output_path_depth,
                 'output_path_svg': output_path_svg,
                 'output_path_csv': output_path_csv
             }
             
             # 确保返回结果中包含标记线数据，优先使用self.projection_data中的标记线
             if 'marker_lines_2d' in self.projection_data:
                 result['marker_lines_2d'] = self.projection_data['marker_lines_2d']
                 print(f"[MAIN_WINDOW] 已将 self.projection_data 中的 {len(self.projection_data['marker_lines_2d'])} 条标记线数据添加到返回结果")
             elif marker_lines_2d is not None:
                 result['marker_lines_2d'] = marker_lines_2d
                 print(f"[MAIN_WINDOW] 已将 {len(marker_lines_2d)} 条标记线数据添加到返回结果")
             else:
                 result['marker_lines_2d'] = []
                 print(f"[MAIN_WINDOW] 未找到标记线数据，返回空列表")
             
             print(f"[MAIN_WINDOW] 投影图像生成完成") 
             # 关闭图形，释放matplotlib资源
             if 'fig' in locals():
                 plt.close(fig)
             return result, None
              
         except Exception as e: 
             print(f"[MAIN_WINDOW] 优化深度图生成失败: {e}") 
             import traceback 
             traceback.print_exc() 
             # 关闭图形 
             if 'fig' in locals(): 
                 plt.close(fig) 
             return None, str(e)

    def generate_2d_depth_image(self, points_2d, depth_values, grid_resolution=0.05, interpolation_method='cubic', colormap='viridis'):
        """生成2D深度图，严格按照以下核心步骤实现：
        1. 读取STL文件三角面片数据（已在投影阶段完成）
        2. 定义投影平面方程（已在拟合阶段完成）
        3. 计算每个面片顶点到平面垂直距离（已在投影阶段完成）
        4. 按分辨率离散化平面并插值生成灰度图
        
        Args:
            points_2d: 2D投影点坐标
            depth_values: 每个点的深度值（到平面的垂直距离）
            grid_resolution: 网格分辨率（mm），默认0.2mm
            interpolation_method: 插值方法（linear, nearest, cubic），默认linear
            colormap: 颜色映射，默认gray_r
        """
        print("[MAIN_WINDOW] 开始生成2D深度图...")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用Agg后端，纯软件渲染，不依赖OpenGL
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
            from scipy.interpolate import griddata
            
            # 检查输入数据
            if points_2d is None or len(points_2d) == 0:
                print("[MAIN_WINDOW] 投影点集为空，无法生成图像")
                QMessageBox.warning(self, "警告", "投影点集为空，无法生成图像")
                return
            
            if len(points_2d.shape) != 2 or points_2d.shape[1] != 2:
                print(f"[MAIN_WINDOW] 投影点集维度错误: {points_2d.shape}")
                QMessageBox.warning(self, "警告", "投影点集维度错误，无法生成图像")
                return
            
            # 确保数据一致性
            min_len = min(len(points_2d), len(depth_values))
            points_2d, depth_values = points_2d[:min_len], depth_values[:min_len]
            print(f"[MAIN_WINDOW] 匹配后点数量: {len(points_2d)}, 深度值数量: {len(depth_values)}")
            
            # 异常点处理：过滤超出均值±3倍标准差的深度值
            print("[MAIN_WINDOW] 开始异常点处理...")
            z_mean, z_std = np.mean(depth_values), np.std(depth_values)
            valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
            filtered_points_2d = points_2d[valid_mask]
            filtered_depth_values = depth_values[valid_mask]
            
            filtered_count = len(points_2d) - len(filtered_points_2d)
            if filtered_count > 0:
                print(f"[MAIN_WINDOW] 已过滤 {filtered_count} 个离群点")
                points_2d = filtered_points_2d
                depth_values = filtered_depth_values
            else:
                print("[MAIN_WINDOW] 未检测到离群点")
            
            # 优化：仅在点数量过多时进行降采样
            if len(points_2d) > 50000:
                print(f"[MAIN_WINDOW] 原始点数量: {len(points_2d)}")
                indices = np.random.choice(len(points_2d), 50000, replace=False)
                points_2d = points_2d[indices]
                depth_values = depth_values[indices]
                print(f"[MAIN_WINDOW] 降采样后点数量: {len(points_2d)}")
            
            # Step 1: 计算投影点范围
            x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
            y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
            print(f"[MAIN_WINDOW] 投影点集范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
            print(f"[MAIN_WINDOW] 创建图形完成，画布大小: 12x10, DPI: 200")
            
            # 添加调试日志，检查Matplotlib版本和可用方法
            print(f"[MAIN_WINDOW] Matplotlib版本: {matplotlib.__version__}")
            print(f"[MAIN_WINDOW] Axes可用属性: {[attr for attr in dir(ax) if 'proj' in attr or 'aspect' in attr]}")
            
            # Step 2: 按分辨率离散化平面并检查内存占用
            xi = np.arange(x_min - 2, x_max + 2, grid_resolution)
            yi = np.arange(y_min - 2, y_max + 2, grid_resolution)
            
            # 计算理论内存占用（每个浮点数据4字节）
            grid_size = len(xi) * len(yi)
            estimated_memory_mb = (grid_size * 4) / (1024 * 1024)  # MB
            max_memory_mb = 500  # 最大允许内存占用
            
            if estimated_memory_mb > max_memory_mb:
                print(f"[MAIN_WINDOW] 警告: 预计内存占用 {estimated_memory_mb:.2f} MB 超过阈值 {max_memory_mb} MB")
                # 自动降低分辨率
                new_resolution = grid_resolution * 2
                print(f"[MAIN_WINDOW] 自动将分辨率降低到 {new_resolution:.1f} mm")
                
                # 重新生成网格
                xi = np.arange(x_min - 2, x_max + 2, new_resolution)
                yi = np.arange(y_min - 2, y_max + 2, new_resolution)
                grid_resolution = new_resolution
                
                # 重新计算内存占用
                grid_size = len(xi) * len(yi)
                estimated_memory_mb = (grid_size * 4) / (1024 * 1024)  # MB
                print(f"[MAIN_WINDOW] 调整后预计内存占用 {estimated_memory_mb:.2f} MB")
            
            xi, yi = np.meshgrid(xi, yi)
            print(f"[MAIN_WINDOW] 平面离散化完成，网格大小: {xi.shape[0]}x{xi.shape[1]}")
            
            # Step 3: 插值生成深度图
            print(f"[MAIN_WINDOW] 开始生成深度图，使用高级插值算法...")
            try:
                # 使用高级插值算法生成深度图
                zi = self.advanced_interpolation(points_2d, depth_values, xi, yi)
            except Exception as e:
                print(f"[MAIN_WINDOW] 高级插值失败，回退到标准插值: {e}")
                # 使用标准插值方法作为备选
                try:
                    # 使用指定插值方法生成深度图
                    zi = griddata(points_2d, depth_values, (xi, yi), method=interpolation_method)
                    
                    # 处理插值失败的点
                    if np.any(np.isnan(zi)):
                        print("[MAIN_WINDOW] 检测到NaN值，使用最近邻插值填充...")
                        nan_mask = np.isnan(zi)
                        zi[nan_mask] = griddata(points_2d, depth_values, 
                                             (xi[nan_mask], yi[nan_mask]), method='nearest')
                except Exception as e:
                    print(f"[MAIN_WINDOW] 线性插值失败，回退到最近邻插值: {e}")
                    zi = griddata(points_2d, depth_values, (xi, yi), method='nearest')
            
            # 添加深度值统计信息
            depth_stats = {
                'min': np.min(depth_values),
                'max': np.max(depth_values),
                'mean': np.mean(depth_values),
                'median': np.median(depth_values),
                'std': np.std(depth_values)
            }
            print(f"[MAIN_WINDOW] 深度值统计: {depth_stats}")
            
            # 计算点云密度
            area = (x_max - x_min) * (y_max - y_min)
            if area > 0:
                density = len(points_2d) / area
            else:
                density = 0
            print(f"[MAIN_WINDOW] 点云密度: {density:.2f} points/mm²")
            
            # 获取深度值范围
            depth_min = np.min(depth_values)
            depth_max = np.max(depth_values)
            print(f"[MAIN_WINDOW] 深度值范围: [{depth_min:.2f}, {depth_max:.2f} mm]")
            
            # 绘制真正的深度图
            print(f"[MAIN_WINDOW] 开始绘制深度图，颜色映射: {colormap}...")
            
            # 绘制深度图
            # 使用xi和yi的实际范围作为extent参数，确保坐标正确匹配
            # 由于应用了深度图增强，使用增强后的数据范围
            enhanced_min = np.min(zi)
            enhanced_max = np.max(zi)
            im = ax.imshow(zi, cmap=colormap, origin='lower', 
                         extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                         vmin=enhanced_min, vmax=enhanced_max, aspect='equal',
                         interpolation='lanczos')  # 使用更平滑的lanczos插值，进一步提高图像连续性
            
            # 添加详细的调试信息
            print(f"[MAIN_WINDOW] 深度图绘制详细信息:")
            print(f"[MAIN_WINDOW]  - 原始深度值范围: [{depth_min:.2f}, {depth_max:.2f}] mm")
            print(f"[MAIN_WINDOW]  - 插值后深度值范围: [{np.nanmin(zi):.2f}, {np.nanmax(zi):.2f}] mm")
            print(f"[MAIN_WINDOW]  - 增强后深度值范围: [{enhanced_min:.4f}, {enhanced_max:.4f}] mm")
            print(f"[MAIN_WINDOW]  - 网格范围(xi): [{xi.min():.2f}, {xi.max():.2f}] mm ({len(xi)}个点)")
            print(f"[MAIN_WINDOW]  - 网格范围(yi): [{yi.min():.2f}, {yi.max():.2f}] mm ({len(yi)}个点)")
            print(f"[MAIN_WINDOW]  - 显示范围(extent): [{xi.min():.2f}, {xi.max():.2f}, {yi.min():.2f}, {yi.max():.2f}]")
            print(f"[MAIN_WINDOW]  - 颜色映射: {colormap}")
            print(f"[MAIN_WINDOW]  - 插值方法: bilinear")
            print(f"[MAIN_WINDOW]  - 坐标原点: lower")
            
            # 保存插值坐标用于联动
            self.projection_data['interpolation_coords'] = (xi, yi)
            self.projection_data['depth_image'] = im
            self.projection_data['depth_ax'] = ax
            self.projection_data['depth_min'] = depth_min
            self.projection_data['depth_max'] = depth_max
            self.projection_data['extent'] = [x_min - 2, x_max + 2, y_min - 2, y_max + 2]
            
            # 添加鼠标点击事件处理，实现深度图与3D模型联动
            def on_depth_image_click(event):
                if event.inaxes != ax:
                    return
                
                # 获取点击的2D坐标
                x_click = event.xdata
                y_click = event.ydata
                
                # 找到离点击位置最近的2D点
                if self.projection_data['points_2d'] is not None:
                    points_2d = self.projection_data['points_2d']
                    distances = np.sqrt((points_2d[:, 0] - x_click)**2 + (points_2d[:, 1] - y_click)**2)
                    min_idx = np.argmin(distances)
                    
                    # 获取对应的3D点
                    projected_points_3d = self.projection_data['projected_points_3d']
                    if min_idx < len(projected_points_3d):
                        clicked_point_3d = projected_points_3d[min_idx]
                        
                        # 在3D视图中高亮显示这个点
                        self.viewer.highlight_point(clicked_point_3d, color=[1, 0, 0], size=3)
                        print(f"[MAIN_WINDOW] 深度图点击位置({x_click:.2f}, {y_click:.2f})对应3D点: {clicked_point_3d}")
            
            # 连接鼠标点击事件
            fig.canvas.mpl_connect('button_press_event', on_depth_image_click)
            
            # 确保颜色映射符合预期：距离越近越亮，越远越暗
            # gray_r 映射：深度值越大（距离越远），颜色越暗；深度值越小（距离越近），颜色越亮
            
            # 添加颜色条，但不显示标签和刻度
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
            cbar.set_label('')
            cbar.set_ticks([])
            
            print("[MAIN_WINDOW] 深度图绘制完成")
            
            # 移除所有文字元素
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # 隐藏网格线和坐标轴刻度
            ax.grid(False)
            ax.tick_params(axis='both', which='both', labelsize=0, length=0)
            
            # 设置视图范围以确保所有内容都可见
            ax.set_xlim(xi.min(), xi.max())
            ax.set_ylim(yi.min(), yi.max())
            ax.autoscale(False)  # 禁用自动缩放，保持正交投影的精确性
            
            # 确保使用正交投影效果
            ax.set_aspect('equal', adjustable='box')  # 保持x和y轴比例一致，实现正交投影效果
            
            # 保存图像为多种格式
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            # 生成时间戳，确保文件名唯一
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式：YYYYMMDD_HHMMSS_mmm
            
            output_path_png = os.path.join(output_dir, f"projection_{timestamp}.png")
            output_path_svg = os.path.join(output_dir, f"projection_{timestamp}.svg")
            output_path_depth = os.path.join(output_dir, f"depth_map_{timestamp}.png")
            output_path_ortho_3d = os.path.join(output_dir, f"ortho_3d_view_{timestamp}.png")
            
            # 保存深度图为高DPI的PNG
            print(f"[MAIN_WINDOW] 准备保存深度图到: {output_path_depth}")
            try:
                print("[MAIN_WINDOW] 开始保存深度图...")
                plt.savefig(output_path_depth, dpi=500, bbox_inches='tight', pad_inches=0.1, format='png', 
                          pil_kwargs={'quality': 95, 'optimize': True}, antialiased=True, rasterized=True)
                print(f"[MAIN_WINDOW] 深度图保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存深度图失败: {e}")
            
            # 保存投影图像为SVG
            print(f"[MAIN_WINDOW] 准备保存投影图像到: {output_path_svg}")
            try:
                print("[MAIN_WINDOW] 开始保存SVG图像...")
                plt.savefig(output_path_svg, format='svg', bbox_inches='tight', pad_inches=0.1)
                print(f"[MAIN_WINDOW] SVG图像保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存SVG图像失败: {e}")
            
            # 获取并保存正交投影的3D效果图
            print(f"[MAIN_WINDOW] 开始生成正交投影的3D效果图...")
            try:
                # 设置视图方向为平面法向量方向
                if self.projection_data.get('plane_normal') is not None:
                    normal = self.projection_data['plane_normal']
                    # 设置相机位置，从平面法向量方向查看
                    self.viewer.plotter.camera.position = normal * 100  # 从距离原点100mm处沿法向量方向查看
                    self.viewer.plotter.camera.focal_point = [0, 0, 0]  # 聚焦于原点
                    self.viewer.plotter.camera.view_up = [0, 1, 0]  # 设置Y轴为上方向
                    
                # 获取正交投影视图
                ortho_image = self.viewer.get_orthographic_view_image(width=1200, height=800)
                
                # 保存正交投影视图
                import cv2
                cv2.imwrite(output_path_ortho_3d, ortho_image)
                print(f"[MAIN_WINDOW] 正交投影3D效果图保存完成: {output_path_ortho_3d}")
            except Exception as e:
                print(f"[MAIN_WINDOW] 生成或保存正交投影3D效果图失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 保存为SVG格式（矢量图，便于后续编辑）
            print(f"[MAIN_WINDOW] 准备保存SVG图像到: {output_path_svg}")
            try:
                print("[MAIN_WINDOW] 开始保存SVG图像...")
                plt.savefig(output_path_svg, format='svg', bbox_inches='tight', pad_inches=0.1)
                print(f"[MAIN_WINDOW] SVG图像保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存SVG图像失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 保存为TIFF格式（无损压缩，适合高精度分析）
            output_path_tiff = os.path.join(output_dir, f"projection_{timestamp}.tiff")
            print(f"[MAIN_WINDOW] 准备保存TIFF图像到: {output_path_tiff}")
            try:
                print("[MAIN_WINDOW] 开始保存TIFF图像...")
                plt.savefig(output_path_tiff, dpi=500, bbox_inches='tight', pad_inches=0.1, format='tiff', 
                          pil_kwargs={'compression': 'tiff_deflate'}, antialiased=True, rasterized=True)
                print(f"[MAIN_WINDOW] TIFF图像保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存TIFF图像失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 保存深度值矩阵为NPY格式（用于进一步数值分析）
            output_path_npy = os.path.join(output_dir, f"depth_matrix_{timestamp}.npy")
            print(f"[MAIN_WINDOW] 准备保存深度矩阵到: {output_path_npy}")
            try:
                print("[MAIN_WINDOW] 开始保存深度矩阵...")
                np.save(output_path_npy, zi)
                print(f"[MAIN_WINDOW] 深度矩阵保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存深度矩阵失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 保存深度值为CSV格式（便于通用软件分析）
            output_path_csv = os.path.join(output_dir, f"depth_values_{timestamp}.csv")
            print(f"[MAIN_WINDOW] 准备保存深度值CSV到: {output_path_csv}")
            try:
                print("[MAIN_WINDOW] 开始保存深度值CSV...")
                # 创建包含2D坐标和对应深度值的结构化数据
                csv_data = np.column_stack((self.projection_data['points_2d'], self.projection_data['depth_values']))
                header = "x,y,depth"
                np.savetxt(output_path_csv, csv_data, delimiter=",", header=header, comments="")
                print(f"[MAIN_WINDOW] 深度值CSV保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存深度值CSV失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 保存投影点云为PLY格式（便于3D软件查看）
            output_path_ply = os.path.join(output_dir, f"projected_points_{timestamp}.ply")
            print(f"[MAIN_WINDOW] 准备保存投影点云到: {output_path_ply}")
            try:
                print("[MAIN_WINDOW] 开始保存投影点云...")
                # 创建Open3D点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.projection_data['projected_points_3d'])
                # 将深度值作为颜色信息（标准化）
                colors = np.zeros_like(self.projection_data['projected_points_3d'])
                depth_values = self.projection_data['depth_values']
                norm_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))
                colors[:, 0] = norm_depth  # 使用红色通道表示深度值
                pcd.colors = o3d.utility.Vector3dVector(colors)
                # 保存PLY文件
                o3d.io.write_point_cloud(output_path_ply, pcd)
                print(f"[MAIN_WINDOW] 投影点云保存完成")
            except Exception as e:
                print(f"[MAIN_WINDOW] 保存投影点云失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 检查图像文件是否成功保存
            # 检查实际保存的文件（depth_map.png 和 projection.svg）是否存在
            depth_map_exists = os.path.exists(output_path_depth) if 'output_path_depth' in locals() else False
            png_exists = os.path.exists(output_path_png) if 'output_path_png' in locals() else False
            svg_exists = os.path.exists(output_path_svg) if 'output_path_svg' in locals() else False
            
            if depth_map_exists or png_exists or svg_exists:
                # 如果任何一个图像文件保存成功，就认为生成成功
                success_path = output_path_depth if depth_map_exists else (output_path_png if png_exists else output_path_svg)
                self.status_label.setText(f"投影图像已生成并保存到: {success_path}")
                self.progress_bar.setValue(100)
                print(f"[MAIN_WINDOW] 投影图像生成完成: {success_path}")
            else:
                self._handle_error("投影图像生成失败，文件未保存", error_type="error", log_level="error")
                plt.close(fig)
                return
            
            # 在界面上展示生成的灰度图
            try:
                self.status_label.setText("正在准备图像展示...")
                self.progress_bar.setValue(90)
                
                # 使用增强型深度图像对话框替代简单对话框
                # 从投影数据中获取标记线
                marker_lines_2d = self.projection_data.get('marker_lines_2d')
                dialog = EnhancedDepthImageDialog(
                    depth_image=depth_image_2d, 
                    extent=extent,
                    marker_lines_2d=marker_lines_2d,
                    title="投影深度图分析", 
                    parent=self
                )
                dialog.show()
                
                # 保存到投影数据中以便后续使用
                self.projection_data['depth_image'] = depth_image_2d
                self.projection_data['extent'] = extent
                self.projection_data['output_path'] = output_path_png
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                
                # 添加一个按钮来关闭对话框
                close_button = QPushButton("关闭")
                close_button.clicked.connect(dialog.close)
                layout.addWidget(close_button)
                
                # 显示对话框
                dialog.exec_()
                
                # 对话框关闭后释放资源
                plt.close(fig)
                print("[MAIN_WINDOW] 释放matplotlib资源完成")
                
            except Exception as e:
                print(f"[MAIN_WINDOW] 在界面上显示图像失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 释放资源
                plt.close(fig)
                print("[MAIN_WINDOW] 释放matplotlib资源完成")
                
                # 回退到原始的外部查看器方式
                reply = QMessageBox.information(self, "投影图像生成完成", 
                                              f"投影图像已成功生成并保存到:\n{output_path_png}\n\n" +
                                              "是否要打开图像查看器查看生成的图像？",
                                              QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # 使用默认图像查看器打开生成的图像
                    import subprocess
                    try:
                        subprocess.Popen([output_path_png], shell=True)
                        print("[MAIN_WINDOW] 使用默认图像查看器打开PNG图像")
                    except Exception as e:
                        print(f"[MAIN_WINDOW] 打开图像查看器失败: {e}")
                    
        except ImportError as e:
            print(f"[MAIN_WINDOW] 缺少必要的库: {e}")
            QMessageBox.critical(self, "错误", f"缺少必要的库: {e}")
        except Exception as e:
            print(f"[MAIN_WINDOW] 生成投影图像失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"生成投影图像失败: {e}")
    
    def on_marked_points_updated(self, points):
        """接收标记点更新信号，更新主窗口状态"""
        self.marked_points = points
        
        # 实现实时预览功能 - 当标记点数量达到1个时开始生成预览
        if len(points) >= 1:
            self.generate_real_time_preview()
        else:
            # 如果标记点少于1个，清除预览
            if hasattr(self, 'preview_dialog') and self.preview_dialog is not None:
                self.preview_dialog.close()
                self.preview_dialog = None
            
        # 根据不同模式显示不同的提示信息和按钮状态
        if hasattr(self, 'model_viewer') and hasattr(self.model_viewer, '_marking_mode'):
            marking_mode = self.model_viewer._marking_mode
            
            if marking_mode == "maxilla":
                # maxilla模式下的多点标记提示
                if len(points) >= 3:
                    self.generate_projection_btn.setEnabled(True)
                    if len(points) == 3:
                        self.status_label.setText(f"已标记 {len(points)} 个点，可生成投影（建议添加更多点以提高精度）")
                    elif len(points) <= 10:
                        self.status_label.setText(f"已标记 {len(points)} 个点，可生成投影（继续添加点以提高精度）")
                    else:
                        self.status_label.setText(f"已标记 {len(points)} 个点，可生成高精度投影")
                else:
                    self.generate_projection_btn.setEnabled(False)
                    remaining = 3 - len(points)
                    self.status_label.setText(f"已标记 {len(points)} 个点，还需标记 {remaining} 个点才能生成投影")
            elif marking_mode == "plane":
                # plane模式下的三点限制提示
                if len(points) >= 3:
                    self.generate_projection_btn.setEnabled(True)
                    self.status_label.setText(f"已标记 {len(points)} 个点，平面已拟合完成")
                else:
                    self.generate_projection_btn.setEnabled(False)
                    remaining = 3 - len(points)
                    self.status_label.setText(f"已标记 {len(points)} 个点，还需标记 {remaining} 个点以拟合平面")
            # 记录日志时检查model_viewer是否存在
            print(f"[MAIN_WINDOW] 标记点更新，当前数量: {len(points)}, 模式: {marking_mode}")
        else:
            # 默认行为
            if len(points) >= 3:
                self.generate_projection_btn.setEnabled(True)
                self.status_label.setText(f"已标记 {len(points)} 个点，可生成投影")
            else:
                self.generate_projection_btn.setEnabled(False)
                remaining = 3 - len(points)
                self.status_label.setText(f"已标记 {len(points)} 个点，还需标记 {remaining} 个点")
            # 记录日志，使用安全的方式
            print(f"[MAIN_WINDOW] 标记点更新，当前数量: {len(points)}, 模式: 未知")
        
        # 当标记点更新时，自动触发投影更新
        self.update_projection_with_markers(points)

    
    def update_projection_with_markers(self, points):
        """当标记点更新时，更新投影图像"""
        print(f"[MAIN_WINDOW] 触发投影更新，当前标记点数量: {len(points)}")
        
        # 确保已生成深度图，并且有深度生成器实例
        if hasattr(self, 'depth_generator') and len(points) >= 2:
            # 如果已有生成的深度图，重新生成包含标记线的版本
            if hasattr(self.depth_generator, 'projection_data') and self.depth_generator.projection_data.get('depth_image') is not None:
                # 获取上颌模型
                maxilla_mesh = self.viewer.get_original_model("maxilla") if hasattr(self.viewer, 'get_original_model') else None
                if maxilla_mesh:
                    # 重新生成深度图，此时会包含标记线投影
                    self.depth_generator.generate_depth_map(maxilla_mesh)
                    print("[MAIN_WINDOW] 深度图已重新生成，包含最新标记线投影")
        
    def generate_real_time_preview(self):
        """生成实时预览"""
        try:
            if not hasattr(self, 'viewer') or not self.viewer.models.get('maxilla'):
                return
            
            # 使用快速方法生成预览数据
            preview_data = self.viewer.generate_preview_projection(self.marked_points)
            if preview_data:
                points_3d, depth_values = preview_data
                
                # 使用预览生成器创建低分辨率预览
                depth_image, extent = self.preview_generator.generate_preview(points_3d, depth_values, grid_resolution=1.0)
                
                # 显示预览
                if depth_image is not None and extent is not None:
                    # 如果已有预览对话框，更新它；否则创建新的
                    if hasattr(self, 'preview_dialog') and self.preview_dialog is not None:
                        try:
                            # 更新现有对话框的数据和显示
                            self.preview_dialog.depth_image = depth_image
                            self.preview_dialog.extent = extent
                            self.preview_dialog.analyzer = DepthAnalyzer(depth_image, extent)
                            
                            # 更新对比度滑块范围
                            min_val = np.min(depth_image)
                            max_val = np.max(depth_image)
                            self.preview_dialog.contrast_min_slider.setRange(int(min_val * 100), int(max_val * 100))
                            self.preview_dialog.contrast_max_slider.setRange(int(min_val * 100), int(max_val * 100))
                            
                            # 应用自动对比度
                            self.preview_dialog.apply_auto_contrast()
                            
                            # 更新显示
                            if self.preview_dialog.view_stack.currentIndex() == 0:
                                self.preview_dialog.update_depth_map()
                            else:
                                self.preview_dialog.update_multi_view()
                        except Exception as update_error:
                            print(f"[MAIN_WINDOW] 更新预览对话框失败: {update_error}")
                            # 如果更新失败，创建新的对话框
                            # 尝试从投影数据中获取标记线用于预览
                            marker_lines_2d = getattr(self, 'projection_data', {}).get('marker_lines_2d')
                            self.preview_dialog = EnhancedDepthImageDialog(depth_image, extent, marker_lines_2d=marker_lines_2d, title="实时预览", parent=self)
                            self.preview_dialog.show()
                    else:
                        # 创建新对话框
                        # 尝试从投影数据中获取标记线用于预览
                        marker_lines_2d = getattr(self, 'projection_data', {}).get('marker_lines_2d')
                        self.preview_dialog = EnhancedDepthImageDialog(depth_image, extent, marker_lines_2d=marker_lines_2d, title="实时预览", parent=self)
                        self.preview_dialog.show()
                    
        except Exception as e:
            print(f"[MAIN_WINDOW] 生成实时预览失败: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止正在运行的线程
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.stop()
        
        # 清理预览生成器缓存
        if hasattr(self, 'preview_generator'):
            self.preview_generator.clear_cache()
        
        # 关闭查看器
        if hasattr(self, 'viewer'):
            # 这里可以添加清理代码
            pass
        
        event.accept()