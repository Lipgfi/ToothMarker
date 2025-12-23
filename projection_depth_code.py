# 投影生成深度图的完整代码实现

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，纯软件渲染，不依赖OpenGL
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import open3d as o3d
from PyQt5.QtWidgets import QMessageBox
import pyvista as pv
import cv2
import time

class ProjectionDepthGenerator:
    """投影生成深度图的核心类"""
    
    def __init__(self, viewer):
        self.viewer = viewer
        self.projection_data = {
            'points_2d': None,
            'projected_points_3d': None,
            'depth_values': None,
            'interpolation_coords': None,
            'depth_image': None,
            'depth_ax': None,
            'depth_min': None,
            'depth_max': None,
            'extent': None,
            'marker_lines_3d': None,
            'marker_lines_2d': None
        }
        self.marker_lines_3d = None
        self.marker_lines_2d = None
    
    def _render_depth_map_direct(self, mesh):
        """直接从3D模型获取深度信息的渲染函数（方法1）
        
        Args:
            mesh: 3D模型对象
            
        Returns:
            tuple: (depth_image, camera_params) 或 None
        """
        try:
            # 检查模型类型并转换为PyVista格式
            if isinstance(mesh, pv.PolyData):
                pv_mesh = mesh
            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'triangles'):
                # 假设是自定义模型格式，转换为PyVista PolyData
                points = np.array(mesh.vertices)
                faces = np.hstack([[3] + list(tri) for tri in mesh.triangles])
                pv_mesh = pv.PolyData(points, faces)
            elif hasattr(mesh, 'points'):  # 可能是PyVista的其他网格类型
                pv_mesh = pv.PolyData(mesh.points)
            elif hasattr(mesh, 'vertex'):  # Open3D格式
                points = np.asarray(mesh.vertex)
                triangles = np.asarray(mesh.triangle)[:, 1:]
                faces = np.hstack([[3] + list(tri) for tri in triangles])
                pv_mesh = pv.PolyData(points, faces)
            elif hasattr(mesh, 'points') and hasattr(mesh, 'triangles'):  # 另一种Open3D格式
                points = np.asarray(mesh.points)
                triangles = np.asarray(mesh.triangles)
                faces = np.hstack([[3] + list(tri) for tri in triangles])
                pv_mesh = pv.PolyData(points, faces)
            else:
                return None
            
            # 统一渲染参数
            PROJECTION_RESOLUTION = (2048, 1536)  # 统一分辨率
            MULTI_SAMPLES = 8  # 统一采样率
            
            # 优化渲染设置
            width, height = PROJECTION_RESOLUTION
            plotter = pv.Plotter(off_screen=True, multi_samples=MULTI_SAMPLES, window_size=[width, height])
            plotter.add_mesh(pv_mesh, color='gray')
            
            # 设置正交投影
            plotter.camera.parallel_projection = True
            
            # 计算合适的相机位置，确保模型完整显示
            bounds = pv_mesh.bounds
            center = pv_mesh.center
            size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            
            # 设置相机位置，从正前方观察
            plotter.camera.position = (center[0], center[1], center[2] + size * 2)
            plotter.camera.focal_point = center
            plotter.camera.view_up = (0, 1, 0)
            plotter.camera.parallel_scale = size / 2
            
            # 渲染并获取深度图
            plotter.render()
            
            # 尝试获取深度图，优先使用更高效的方法
            depth_img = None
            try:
                # 优先使用PyVista内置方法，更高效
                depth_img = plotter.get_depth_image()
                # 反转Y轴以匹配图像坐标
                depth_img = np.flipud(depth_img)
            except (AttributeError, TypeError):
                # 备选方案：使用VTK直接获取深度缓冲区
                try:
                    import vtk
                    ren_win = plotter.ren_win
                    vtk_width, vtk_height = ren_win.GetSize()
                    
                    depth_buffer = vtk.vtkFloatArray()
                    depth_buffer.SetNumberOfComponents(1)
                    depth_buffer.SetNumberOfTuples(vtk_width * vtk_height)
                    
                    ren_win.GetZbufferData(0, 0, vtk_width-1, vtk_height-1, depth_buffer)
                    
                    depth_data = np.array(depth_buffer)
                    depth_img = depth_data.reshape((vtk_height, vtk_width))
                except Exception as e:
                    print(f"[深度渲染] VTK获取深度缓冲区失败: {e}")
                    return None
            
            if depth_img is None:
                print("[深度渲染] 无法获取深度图像")
                return None
            
            # 获取相机参数
            camera_params = {
                'width': width,
                'height': height,
                'position': tuple(plotter.camera.position),
                'focal_point': tuple(plotter.camera.focal_point),
                'view_up': tuple(plotter.camera.view_up),
                'parallel_projection': plotter.camera.parallel_projection,
                'parallel_scale': plotter.camera.parallel_scale,
                'type': 'direct_rendering'
            }
            
            plotter.close()
            return depth_img, camera_params
            
        except Exception as e:
            print(f"[深度渲染] 渲染失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def project_to_plane(self, mesh, plane_eq, model_type="maxilla"):
        """
        将3D模型投影到指定平面上
        
        Args:
            mesh: 要投影的3D模型
            plane_eq: 平面方程 [a, b, c, d]，满足 ax + by + cz + d = 0
            model_type: 模型类型（"maxilla" 或 "mandible"）
            
        Returns:
            投影后的2D点集和深度图像
        """
        try:
            print(f"[{model_type}平面投影] 开始将模型投影到平面: {plane_eq}")
            
            # 转换为PyVista格式（如需要）
            if isinstance(mesh, pv.PolyData):
                pv_mesh = mesh
            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'triangles'):
                # 假设是自定义模型格式，转换为PyVista PolyData
                points = np.array(mesh.vertices)
                faces = np.hstack([[3] + list(tri) for tri in mesh.triangles])
                pv_mesh = pv.PolyData(points, faces)
            elif hasattr(mesh, 'points') and hasattr(mesh, 'triangles'):  # Open3D格式
                points = np.asarray(mesh.points)
                triangles = np.asarray(mesh.triangles)
                faces = np.hstack([[3] + list(tri) for tri in triangles])
                pv_mesh = pv.PolyData(points, faces)
            elif hasattr(mesh, 'points'):  # 可能是PyVista的其他网格类型
                pv_mesh = pv.PolyData(mesh.points)
            elif hasattr(mesh, 'vertex'):  # 另一种Open3D格式
                points = np.asarray(mesh.vertex)
                triangles = np.asarray(mesh.triangle)[:, 1:]
                faces = np.hstack([[3] + list(tri) for tri in triangles])
                pv_mesh = pv.PolyData(points, faces)
            else:
                print(f"[{model_type}平面投影] 不支持的模型类型: {type(mesh)}")
                return None
            
            # 统一渲染参数
            PROJECTION_RESOLUTION = (2048, 1536)  # 统一分辨率
            MULTI_SAMPLES = 8  # 统一采样率
            
            # 平面参数
            a, b, c, d = plane_eq
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)  # 归一化法向量
            
            # 创建与目标平面平行的相机平面
            center = pv_mesh.center
            # 计算相机位置（沿平面法向量方向远离模型）
            camera_pos = center + normal * np.linalg.norm(pv_mesh.bounds) * 2
            
            # 设置投影相机
            plotter = pv.Plotter(off_screen=True, multi_samples=MULTI_SAMPLES, window_size=PROJECTION_RESOLUTION)
            plotter.add_mesh(pv_mesh, color='gray')
            
            # 设置正交投影，方向沿平面法向量
            plotter.camera.parallel_projection = True
            plotter.camera.position = camera_pos
            plotter.camera.focal_point = center
            # 计算合适的上方向向量
            view_up = np.cross(normal, [0, 1, 0]) if np.linalg.norm(np.cross(normal, [0, 1, 0])) > 1e-6 else [1, 0, 0]
            plotter.camera.view_up = view_up
            plotter.camera.parallel_scale = np.max(pv_mesh.bounds[1::2] - pv_mesh.bounds[::2]) / 2
            
            # 渲染并获取深度图
            plotter.render()
            
            # 获取深度图
            depth_img = None
            try:
                # 直接使用VTK获取深度缓冲区
                ren_win = plotter.ren_win
                vtk_width, vtk_height = ren_win.GetSize()
                
                import vtk
                depth_buffer = vtk.vtkFloatArray()
                depth_buffer.SetNumberOfComponents(1)
                depth_buffer.SetNumberOfTuples(vtk_width * vtk_height)
                
                ren_win.GetZbufferData(0, 0, vtk_width-1, vtk_height-1, depth_buffer)
                depth_data = np.array(depth_buffer)
                depth_img = depth_data.reshape((vtk_height, vtk_width))
                print(f"[{model_type}平面投影] 成功使用VTK直接获取深度缓冲区")
            except Exception as e:
                print(f"[{model_type}平面投影] VTK获取深度缓冲区失败: {e}")
                return None
            
            if depth_img is None:
                print(f"[{model_type}平面投影] 无法获取深度图像")
                return None
            
            # 转换为灰度图
            depth_normalized = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img) + 1e-8)
            depth_gray = (255 * (1 - depth_normalized)).astype(np.uint8)
            
            plotter.close()
            return depth_gray
            
        except Exception as e:
            print(f"[{model_type}平面投影] 投影失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_depth_map(self, mesh=None, model_type="maxilla"):
        """生成表示3D模型深度信息的灰度图像
        
        Args:
            mesh: 3D模型对象，如果为None则从viewer获取
            model_type: 模型类型（"maxilla" 或 "mandible"）
            
        Returns:
            生成的深度图像或None
        """
        try:
            # 获取模型
            if mesh is None:
                if hasattr(self.viewer, 'get_models') and model_type in self.viewer.get_models():
                    mesh = self.viewer.get_original_model(model_type)
                    print(f"[{model_type}深度图生成] 从viewer获取模型成功")
                else:
                    print(f"[{model_type}深度图生成] 无法从viewer获取{model_type}模型")
                    return None
            
            # 检查模型是否有效
            if mesh is None:
                print(f"[{model_type}深度图生成] 模型为None")
                return None
            
            # 使用直接渲染方法生成深度图
            result = self._render_depth_map_direct(mesh)
            if result is not None:
                depth_img, camera_params = result
                print(f"[{model_type}深度图生成] 深度图生成成功，尺寸: {depth_img.shape}")
                return depth_img
            else:
                print(f"[{model_type}深度图生成] 深度图生成失败")
                return None
                
        except Exception as e:
            print(f"[{model_type}深度图生成] 生成深度图失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        """生成表示3D模型深度信息的灰度图像
        
        Args:
            maxilla_mesh: 可选，要处理的3D模型，应为pyvista.PolyData类型
            
        Returns:
            tuple: (depth_gray_image, camera_params, depth_data) 或 None
        """
        print("[深度图生成] 开始生成深度图...")
        start_time = time.time()
        
        try:
            # 检查并准备模型
            if maxilla_mesh is None:
                if hasattr(self.viewer, 'get_models') and 'maxilla' in self.viewer.get_models():
                    maxilla_mesh = self.viewer.get_original_model("maxilla")
                    print("[深度图生成] 从viewer获取上颌模型")
                else:
                    print("[深度图生成] 警告: 请先加载上颌模型")
                    return None
            
            # 验证模型类型
            if not isinstance(maxilla_mesh, pv.PolyData):
                print(f"[深度图生成] 警告: 模型类型不正确，期望pyvista.PolyData，实际为{type(maxilla_mesh)}")
                return None
            
            # 方法1：使用专门的深度图渲染函数直接获取深度信息
            result = self._render_depth_map_direct(maxilla_mesh)
            
            if result is not None:
                depth_img, camera_params = result
                
                # 转换深度图为灰度图像（0-255）
                # 归一化深度值到0-1范围
                min_depth = np.min(depth_img)
                max_depth = np.max(depth_img)
                
                if max_depth > min_depth:
                    depth_normalized = (depth_img - min_depth) / (max_depth - min_depth)
                else:
                    depth_normalized = depth_img
                
                # 转换为灰度图像（0-255）
                depth_gray = (255 * (1 - depth_normalized)).astype(np.uint8)  # 反转，近色深显示为白色
                
                print(f"[深度图生成] 深度图生成成功: 大小 {depth_gray.shape}, 深度范围 [{min_depth:.3f}, {max_depth:.3f}]")
                elapsed_time = time.time() - start_time
                print(f"[深度图生成] 耗时: {elapsed_time:.2f}秒")
                
                # 在生成深度图后添加投影线绘制
                # 无论viewer.marker_lines是否存在，都调用_get_marker_lines_from_viewer()获取标记线数据
                self._get_marker_lines_from_viewer()
                
                # 获取平面方程
                if hasattr(self.viewer, 'plane_params') and self.viewer.plane_params:
                    plane_eq = self.viewer.plane_params
                    
                    # 计算并添加两个平面的交线到标记线列表
                    if hasattr(self.viewer, 'second_plane_params') and self.viewer.second_plane_params:
                        # 计算两个平面的交线
                        plane1 = plane_eq
                        plane2 = self.viewer.second_plane_params
                        
                        # 平面方程：ax + by + cz + d = 0
                        a1, b1, c1, d1 = plane1
                        a2, b2, c2, d2 = plane2
                        
                        # 计算交线的方向向量
                        direction = np.cross([a1, b1, c1], [a2, b2, c2])
                        
                        # 检查两个平面是否平行
                        if np.linalg.norm(direction) < 1e-6:
                            print("[深度图生成] 两个平面平行，没有交线")
                        else:
                            # 计算交线上的一个点
                            # 选择z=0平面与交线的交点
                            det = a1 * b2 - a2 * b1
                            if abs(det) < 1e-6:
                                # 用y=0平面
                                det = a1 * c2 - a2 * c1
                                if abs(det) < 1e-6:
                                    # 用x=0平面
                                    det = b1 * c2 - b2 * c1
                                    x = 0
                                    y = (b2 * d1 - b1 * d2) / det
                                    z = (c1 * d2 - c2 * d1) / det
                                else:
                                    x = (a2 * d1 - a1 * d2) / det
                                    y = 0
                                    z = (c1 * d2 - c2 * d1) / det
                            else:
                                x = (b2 * d1 - b1 * d2) / det
                                y = (a1 * d2 - a2 * d1) / det
                                z = 0
                            
                            point = np.array([x, y, z])
                            
                            # 计算交线的长度，使用模型的对角线长度
                            bounds = maxilla_mesh.bounds
                            x_span = bounds[1] - bounds[0]
                            y_span = bounds[3] - bounds[2]
                            z_span = bounds[5] - bounds[4]
                            line_length = np.sqrt(x_span**2 + y_span**2 + z_span**2) * 2
                            
                            # 计算交线的两个端点
                            point1 = point + direction * line_length
                            point2 = point - direction * line_length
                            
                            # 将交线添加到标记线列表
                            intersection_line = np.array([point1, point2])
                            self.marker_lines_3d.append(intersection_line)
                            self.marker_lines_info.append("plane_intersection")
                            print("[深度图生成] 计算并添加了两个平面的交线")
                    
                    # 创建模型加载器实例用于投影计算
                    from processing.model_loader import ModelLoader
                    model_loader = ModelLoader()
                    
                    # 转换深度图为彩色图像以便绘制彩色线条
                    if len(depth_gray.shape) == 2:
                        depth_gray = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                    
                    # 获取深度图尺寸
                    height, width = depth_gray.shape[:2]
                    
                    # 对每条标记线进行投影并绘制
                    if hasattr(self, 'marker_lines_3d') and self.marker_lines_3d:
                        print(f"[深度图生成] 开始绘制{len(self.marker_lines_3d)}条标记线")
                        for i, line_3d in enumerate(self.marker_lines_3d):
                            try:
                                line_type = self.marker_lines_info[i] if hasattr(self, 'marker_lines_info') and i < len(self.marker_lines_info) else "other"
                                
                                # 确保line_3d是numpy数组
                                line_3d = np.array(line_3d)
                                if line_3d.shape[0] >= 2:
                                    # 投影3D线到平面
                                    # 调整线数据格式以匹配project_line_to_plane方法的要求
                                    line = [line_3d[0], line_3d[1], line_type]
                                    projected_line = model_loader.project_line_to_plane(line, plane_eq)
                                    
                                    if projected_line:
                                        (p1_proj_3d, p2_proj_3d, mode) = projected_line
                                        
                                        # 使用相机参数将3D平面坐标映射到2D图像坐标
                                        # 这里简化处理，使用相机参数中的范围信息
                                        if 'extent' in camera_params:
                                            x_min, x_max, y_min, y_max = camera_params['extent']
                                        else:
                                            # 如果没有extent信息，使用模型的包围盒
                                            bounds = maxilla_mesh.bounds
                                            x_min, x_max = bounds[0], bounds[1]
                                            y_min, y_max = bounds[2], bounds[3]
                                        
                                        # 计算x和y方向的缩放因子
                                        x_scale = width / (x_max - x_min)
                                        y_scale = height / (y_max - y_min)
                                        
                                        # 将3D投影点转换为2D图像坐标
                                        x1 = int((p1_proj_3d[0] - x_min) * x_scale)
                                        y1 = int((p1_proj_3d[1] - y_min) * y_scale)
                                        x2 = int((p2_proj_3d[0] - x_min) * x_scale)
                                        y2 = int((p2_proj_3d[1] - y_min) * y_scale)
                                        
                                        # 确保坐标在图像范围内
                                        x1 = max(0, min(x1, width-1))
                                        y1 = max(0, min(y1, height-1))
                                        x2 = max(0, min(x2, width-1))
                                        y2 = max(0, min(y2, height-1))
                                        
                                        # 根据线类型设置颜色
                                        if line_type == "maxilla":
                                            color = (128, 0, 128)  # 紫色（BGR格式）
                                            thickness = 3  # 增加线宽以确保可见性
                                            print(f"[深度图生成] 绘制第{i}条线，类型: maxilla，颜色: 紫色，坐标: ({x1},{y1})-({x2},{y2})")
                                        elif line_type == "plane_intersection":
                                            color = (0, 0, 255)  # 平面交线使用红色（BGR格式）
                                            thickness = 3  # 增加线宽以确保可见性
                                            print(f"[深度图生成] 绘制第{i}条线，类型: plane_intersection，颜色: 红色，坐标: ({x1},{y1})-({x2},{y2})")
                                        else:
                                            color = (255, 0, 0)  # 其他类型使用蓝色
                                            thickness = 2
                                            print(f"[深度图生成] 绘制第{i}条线，类型: {line_type}，颜色: 蓝色，坐标: ({x1},{y1})-({x2},{y2})")
                                        
                                        # 绘制线条
                                        cv2.line(depth_gray, (x1, y1), (x2, y2), color, thickness)
                            except Exception as e:
                                print(f"[深度图生成] 绘制第{i}条标记线时出错: {e}")
                                import traceback
                                traceback.print_exc()
                
                # 将深度图保存为PNG文件
                output_path = 'generated_depth_map.png'
                cv2.imwrite(output_path, depth_gray)
                print(f"[深度图生成] 深度图已保存为 {output_path}")
                
                return depth_gray, camera_params, depth_img
            
            # 方法2：如果方法1失败，使用XY投影图像转换为灰度图
            print("[深度图生成] 直接渲染失败，尝试使用XY投影图像转换...")
            
            # 检查是否有XY投影图像可用
            if hasattr(self, 'projection_data') and 'color_image' in self.projection_data:
                color_img = self.projection_data['color_image']
                
                # 转换为BGR格式以便后续处理
                color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                
                # 添加标记线投影到彩色图像
                # 无论viewer.marker_lines是否存在，都调用_get_marker_lines_from_viewer()获取标记线数据
                self._get_marker_lines_from_viewer()
                
                # 获取平面方程
                if hasattr(self.viewer, 'plane_params') and self.viewer.plane_params:
                    plane_eq = self.viewer.plane_params
                    
                    # 创建模型加载器实例用于投影计算
                    from processing.model_loader import ModelLoader
                    model_loader = ModelLoader()
                    
                    # 获取深度图尺寸
                    height, width = color_img_bgr.shape[:2]
                    
                    # 对每条标记线进行投影并绘制
                    if hasattr(self, 'marker_lines_3d') and self.marker_lines_3d:
                        for i, line_3d in enumerate(self.marker_lines_3d):
                            line_type = self.marker_lines_info[i] if hasattr(self, 'marker_lines_info') and i < len(self.marker_lines_info) else "other"
                            
                            # 确保line_3d是numpy数组
                            line_3d = np.array(line_3d)
                            if line_3d.shape[0] >= 2:
                                # 投影3D线到平面
                                # 调整线数据格式以匹配project_line_to_plane方法的要求
                                line = [line_3d[0], line_3d[1], line_type]
                                projected_line = model_loader.project_line_to_plane(line, plane_eq)
                                
                                if projected_line:
                                    (p1_proj_3d, p2_proj_3d, mode) = projected_line
                                    
                                    # 使用模型的包围盒将3D平面坐标映射到2D图像坐标
                                    bounds = maxilla_mesh.bounds
                                    x_min, x_max = bounds[0], bounds[1]
                                    y_min, y_max = bounds[2], bounds[3]
                                    
                                    # 计算x和y方向的缩放因子
                                    x_scale = width / (x_max - x_min)
                                    y_scale = height / (y_max - y_min)
                                    
                                    # 将3D投影点转换为2D图像坐标
                                    x1 = int((p1_proj_3d[0] - x_min) * x_scale)
                                    y1 = int((p1_proj_3d[1] - y_min) * y_scale)
                                    x2 = int((p2_proj_3d[0] - x_min) * x_scale)
                                    y2 = int((p2_proj_3d[1] - y_min) * y_scale)
                                    
                                    # 确保坐标在图像范围内
                                    x1 = max(0, min(x1, width-1))
                                    y1 = max(0, min(y1, height-1))
                                    x2 = max(0, min(x2, width-1))
                                    y2 = max(0, min(y2, height-1))
                                    
                                    # 绘制紫色线条（BGR格式，紫色为(128, 0, 128)）
                                    color = (128, 0, 128)  # 紫色
                                    cv2.line(color_img_bgr, (x1, y1), (x2, y2), color, 2)
                                    
                                    print(f"[深度图生成] 绘制了{line_type}类型的紫色线条")
            
                # 使用OpenCV将BGR转换为灰度图
                depth_gray = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2GRAY)
                
                print(f"[深度图生成] 成功使用XY投影图像转换为灰度图，大小: {depth_gray.shape}")
                elapsed_time = time.time() - start_time
                print(f"[深度图生成] 耗时: {elapsed_time:.2f}秒")
                
                # 构造相机参数
                camera_params = {
                    'width': depth_gray.shape[1],
                    'height': depth_gray.shape[0],
                    'type': 'xy_projection'
                }
                
                # 在生成深度图后再次添加投影线绘制（冗余保障，确保线条被绘制）
                # 直接使用之前获取的标记线数据，无需再次调用_get_marker_lines_from_viewer()
                if hasattr(self.viewer, 'plane_params') and self.viewer.plane_params and hasattr(self, 'marker_lines_3d') and self.marker_lines_3d:
                    plane_eq = self.viewer.plane_params
                    
                    # 创建模型加载器实例用于投影计算
                    from processing.model_loader import ModelLoader
                    model_loader = ModelLoader()
                    
                    # 转换深度图为彩色图像以便绘制彩色线条
                    if len(depth_gray.shape) == 2:
                        depth_gray = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                    
                    # 获取深度图尺寸
                    height, width = depth_gray.shape[:2]
                    
                    # 对每条标记线进行投影并绘制
                    for i, line_3d in enumerate(self.marker_lines_3d):
                        line_type = self.marker_lines_info[i] if hasattr(self, 'marker_lines_info') and i < len(self.marker_lines_info) else "other"
                        
                        # 确保line_3d是numpy数组
                        line_3d = np.array(line_3d)
                        if line_3d.shape[0] >= 2:
                            # 投影3D线到平面
                            # 调整线数据格式以匹配project_line_to_plane方法的要求
                            line = [line_3d[0], line_3d[1], line_type]
                            projected_line = model_loader.project_line_to_plane(line, plane_eq)
                            
                            if projected_line:
                                (p1_proj_3d, p2_proj_3d, mode) = projected_line
                                
                                # 使用相机参数将3D平面坐标映射到2D图像坐标
                                # 这里简化处理，使用模型的包围盒
                                bounds = maxilla_mesh.bounds
                                x_min, x_max = bounds[0], bounds[1]
                                y_min, y_max = bounds[2], bounds[3]
                                
                                # 计算x和y方向的缩放因子
                                x_scale = width / (x_max - x_min)
                                y_scale = height / (y_max - y_min)
                                
                                # 将3D投影点转换为2D图像坐标
                                x1 = int((p1_proj_3d[0] - x_min) * x_scale)
                                y1 = int((p1_proj_3d[1] - y_min) * y_scale)
                                x2 = int((p2_proj_3d[0] - x_min) * x_scale)
                                y2 = int((p2_proj_3d[1] - y_min) * y_scale)
                                
                                # 确保坐标在图像范围内
                                x1 = max(0, min(x1, width-1))
                                y1 = max(0, min(y1, height-1))
                                x2 = max(0, min(x2, width-1))
                                y2 = max(0, min(y2, height-1))
                                
                                # 绘制紫色线条（BGR格式，紫色为(128, 0, 128)）
                                color = (128, 0, 128)  # 紫色
                                cv2.line(depth_gray, (x1, y1), (x2, y2), color, 2)
                                
                                print(f"[深度图生成] 绘制了{line_type}类型的紫色线条")
                
                # 将深度图保存为PNG文件
                output_path = 'generated_depth_map_xy.png'
                cv2.imwrite(output_path, depth_gray)
                print(f"[深度图生成] XY投影深度图已保存为 {output_path}")
                
                return depth_gray, camera_params, None
            
            print("[深度图生成] 所有深度图生成方法均失败")
            return None
            
        except Exception as e:
            print(f"[深度图生成] 深度图生成错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_open3d_depth_image(self, maxilla_mesh=None):
        """使用Open3D的可视化界面手动捕获深度图
        该方法调用ModelViewer中的Open3D可视化功能，允许用户手动调整视角
        按下'C'键捕获当前视角的深度图
        
        Args:
            maxilla_mesh: 可选，要显示的上颌模型
            
        Returns:
            tuple: (color_image, depth_image, camera_params) 或 None
        """
        import logging
        logger = logging.getLogger('ProjectionDepthGenerator')
        
        print("[深度图修复] 开始使用Open3D手动捕获深度图...")
        logger.info("开始Open3D深度图捕获流程")
        
        try:
            import time
            start_time = time.time()
            
            # 验证viewer对象
            if not hasattr(self, 'viewer') or self.viewer is None:
                error_msg = "[深度图修复] 错误: viewer对象不存在或为None"
                print(error_msg)
                logger.error(error_msg)
                QMessageBox.critical(None, "错误", "深度图生成器初始化失败: 找不到视图组件")
                # 计算总耗时
                elapsed_time = time.time() - start_time
                print(f"[投影生成] 投影过程完成，总耗时: {elapsed_time:.2f}秒")
                return None
            
            # 验证viewer是否有capture_depth_image_with_open3d方法
            if not hasattr(self.viewer, 'capture_depth_image_with_open3d'):
                error_msg = "[深度图修复] 错误: viewer对象没有capture_depth_image_with_open3d方法"
                print(error_msg)
                logger.error(error_msg)
                QMessageBox.critical(None, "错误", "深度图生成器初始化失败: 找不到视图组件")
                return None
            
            # 记录当前工作目录和环境信息
            env_info = f"[深度图修复] 当前工作目录: {os.getcwd()}"
            print(env_info)
            logger.debug(env_info)
            
            version_info = f"[深度图修复] Open3D版本: {o3d.__version__}"
            print(version_info)
            logger.debug(version_info)
            
            # 检查并准备模型
            if maxilla_mesh is None:
                # 尝试从viewer获取模型
                if hasattr(self.viewer, 'get_models') and 'maxilla' in self.viewer.get_models():
                    maxilla_mesh = self.viewer.get_original_model("maxilla")
                    model_info = "[深度图修复] 从viewer获取上颌模型"
                    print(model_info)
                    logger.info(model_info)
                else:
                    error_msg = "[深度图修复] 错误: 找不到上颌模型"
                    print(error_msg)
                    logger.error(error_msg)
                    QMessageBox.warning(None, "警告", "请先加载上颌模型")
                    return None
            
            # 记录模型信息
            if hasattr(maxilla_mesh, 'vertices'):
                model_details = f"[深度图修复] 模型信息: 顶点数量={len(maxilla_mesh.vertices)}, 类型={type(maxilla_mesh)}"
            elif hasattr(maxilla_mesh, 'points'):
                model_details = f"[深度图修复] 模型信息: 点数量={len(maxilla_mesh.points)}, 类型={type(maxilla_mesh)}"
            else:
                model_details = f"[深度图修复] 模型信息: 类型={type(maxilla_mesh)}"
            print(model_details)
            logger.debug(model_details)
            
            method_call_info = "[深度图修复] 调用ModelViewer.capture_depth_image_with_open3d方法..."
            print(method_call_info)
            logger.info(method_call_info)
            
            # 调用ModelViewer中的Open3D深度图捕获功能
            result = self.viewer.capture_depth_image_with_open3d(maxilla_mesh)
            
            elapsed_time = time.time() - start_time
            time_info = f"[深度图修复] 深度图捕获操作耗时: {elapsed_time:.2f}秒"
            print(time_info)
            logger.info(time_info)
            
            if result is not None:
                try:
                    # 验证结果元组结构
                    if not isinstance(result, tuple) or len(result) != 3:
                        error_msg = f"[深度图修复] 错误: 返回结果结构不正确 - 类型={type(result)}, 长度={len(result) if isinstance(result, (list, tuple)) else '非序列'}"
                        print(error_msg)
                        logger.error(error_msg)
                        QMessageBox.warning(None, "警告", f"深度图数据格式错误: 返回结果结构不正确")
                        # 尝试使用备用方法
                        print("[深度图修复] 数据格式不正确，尝试使用PyVista备用方法...")
                        return self.generate_depth_image_with_pyvista(maxilla_mesh)
                    
                    color_img, depth_img, camera_params = result
                    
                    # 验证返回数据的有效性
                    if color_img is None or depth_img is None:
                        warning_msg = f"[深度图修复] 警告: 返回的图像数据为空 - color_img={color_img is not None}, depth_img={depth_img is not None}"
                        print(warning_msg)
                        logger.warning(warning_msg)
                        QMessageBox.warning(None, "警告", "深度图捕获返回了空数据")
                        # 尝试使用备用方法
                        print("[深度图修复] 图像数据为空，尝试使用PyVista备用方法...")
                        return self.generate_depth_image_with_pyvista(maxilla_mesh)
                    
                    # 保存到投影数据中
                    self.projection_data['open3d_color_image'] = color_img
                    self.projection_data['open3d_depth_image'] = depth_img
                    self.projection_data['open3d_camera_params'] = camera_params
                    
                    # 尝试获取图像形状信息
                    try:
                        shape_info = f"[深度图修复] Open3D深度图捕获成功，颜色图大小: {color_img.shape}，深度图大小: {depth_img.shape}"
                    except Exception:
                        shape_info = "[深度图修复] Open3D深度图捕获成功"
                    print(shape_info)
                    logger.info(shape_info)
                    
                    # 尝试记录相机参数
                    try:
                        if camera_params is not None:
                            camera_info = f"[深度图修复] 相机参数: 类型={type(camera_params)}"
                            logger.debug(camera_info)
                    except Exception as cam_e:
                        logger.warning(f"[深度图修复] 记录相机参数时出错: {cam_e}")
                    
                    # 显示成功消息
                    QMessageBox.information(None, "成功", "深度图捕获成功！\n请按C键确认并继续操作。")
                    
                    return result
                except ValueError as e:
                    error_msg = f"[深度图修复] 错误: 返回结果格式不正确: {e}"
                    print(error_msg)
                    logger.error(error_msg)
                    QMessageBox.warning(None, "警告", f"深度图数据格式错误: {str(e)}")
                    # 尝试使用备用方法
                    print("[深度图修复] 返回结果格式错误，尝试使用PyVista备用方法...")
                    return self.generate_depth_image_with_pyvista(maxilla_mesh)
            else:
                warning_msg = "[深度图修复] Open3D深度图捕获失败"
                print(warning_msg)
                logger.warning(warning_msg)
                # 显示详细的错误消息和操作指导，并提供使用备用方法的选项
                reply = QMessageBox.warning(
                    None, 
                    "Open3D深度图捕获失败", 
                    "深度图捕获过程中出现问题。\n\n"+
                    "请尝试以下解决方法:\n"+
                    "1. 确保已正确加载3D模型\n"+
                    "2. 检查OpenGL是否正常工作\n"+
                    "3. 按C键尝试手动捕获\n"+
                    "4. 使用PyVista备用投影方法",
                    QMessageBox.Retry | QMessageBox.Yes | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Yes:
                    print("[深度图修复] 用户选择使用PyVista备用方法...")
                    return self.generate_depth_image_with_pyvista(maxilla_mesh)
                elif reply == QMessageBox.Retry:
                    print("[深度图修复] 用户选择重试Open3D方法...")
                    return self.generate_open3d_depth_image(maxilla_mesh)
                else:
                    print("[深度图修复] 用户取消操作...")
                    return None
                
        except Exception as e:
            error_msg = f"[深度图修复] 捕获深度图时发生异常: {e}"
            print(error_msg)
            logger.error(error_msg, exc_info=True)
            
            import traceback
            traceback_info = traceback.format_exc()
            print(f"[深度图修复] 错误详情: {traceback_info}")
            logger.debug(f"[深度图修复] 完整错误堆栈: {traceback_info}")
            
            # 根据异常类型提供更具体的错误消息
            if "OpenGL" in str(e) or "WGL" in str(e):
                user_msg = f"深度图捕获过程中发生错误:\n图形渲染错误。请尝试更新显卡驱动或重启应用程序。\n\n详细错误: {str(e)}"
            elif "context" in str(e).lower():
                user_msg = f"深度图捕获过程中发生错误:\n上下文初始化错误。请确保系统支持OpenGL 3.3或更高版本。\n\n详细错误: {str(e)}"
            else:
                user_msg = f"深度图捕获过程中发生错误:\n{str(e)}\n\n请检查Open3D是否正确安装，以及系统是否支持OpenGL渲染。"
                
            # 询问用户是否使用备用方法
            reply = QMessageBox.critical(
                None, 
                "错误", 
                user_msg + "\n\n是否尝试使用PyVista备用方法？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                print("[深度图修复] 异常后尝试使用PyVista备用方法...")
                return self.generate_depth_image_with_pyvista(maxilla_mesh)
            
            return None
            
    def generate_depth_image_with_pyvista(self, maxilla_mesh=None):
        """使用PyVista作为备用方法生成深度图
        该方法不依赖Open3D，使用PyVista直接进行投影计算，更稳定但交互性较差
        
        Args:
            maxilla_mesh: 可选，要显示的上颌模型
            
        Returns:
            tuple: (color_image, depth_image, camera_params) 或 None
        """
        import logging
        logger = logging.getLogger('ProjectionDepthGenerator')
        
        print("[深度图修复] 开始使用PyVista备用方法生成深度图...")
        logger.info("开始PyVista深度图生成流程")
        
        try:
            import time
            import pyvista as pv
            start_time = time.time()
            
            # 验证viewer对象
            if not hasattr(self, 'viewer') or self.viewer is None:
                error_msg = "[深度图修复] 错误: viewer对象不存在或为None"
                print(error_msg)
                logger.error(error_msg)
                QMessageBox.critical(None, "错误", "深度图生成器初始化失败: 找不到视图组件")
                return None
            
            # 检查并准备模型
            if maxilla_mesh is None:
                # 尝试从viewer获取模型
                if hasattr(self.viewer, 'get_models') and 'maxilla' in self.viewer.get_models():
                    maxilla_mesh = self.viewer.get_original_model("maxilla")
                    model_info = "[深度图修复] 从viewer获取上颌模型"
                    print(model_info)
                    logger.info(model_info)
                else:
                    error_msg = "[深度图修复] 错误: 找不到上颌模型"
                    print(error_msg)
                    logger.error(error_msg)
                    QMessageBox.warning(None, "警告", "请先加载上颌模型")
                    return None
            
            # 记录模型信息
            if hasattr(maxilla_mesh, 'vertices'):
                model_details = f"[深度图修复] 模型信息: 顶点数量={len(maxilla_mesh.vertices)}, 类型={type(maxilla_mesh)}"
            elif hasattr(maxilla_mesh, 'points'):
                model_details = f"[深度图修复] 模型信息: 点数量={len(maxilla_mesh.points)}, 类型={type(maxilla_mesh)}"
            else:
                model_details = f"[深度图修复] 模型信息: 类型={type(maxilla_mesh)}"
            print(model_details)
            logger.debug(model_details)
            
            # 创建PyVista渲染器
            plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
            
            # 检查并转换模型类型
            if isinstance(maxilla_mesh, pv.DataSet):
                plotter.add_mesh(maxilla_mesh, color='white')
                print("[深度图修复] 已添加PyVista格式模型")
            else:
                try:
                    # 尝试将其他格式转换为PyVista格式
                    if hasattr(maxilla_mesh, 'vertices') and hasattr(maxilla_mesh, 'triangles'):
                        # 假设是类似trimesh的格式
                        vertices = np.array(maxilla_mesh.vertices)
                        triangles = np.array(maxilla_mesh.triangles).reshape(-1, 3)
                        pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(triangles), 1), 3), triangles]))
                        plotter.add_mesh(pv_mesh, color='white')
                        print("[深度图修复] 已转换并添加模型")
                    else:
                        raise TypeError("不支持的模型格式")
                except Exception as e:
                    error_msg = f"[深度图修复] 模型转换错误: {e}"
                    print(error_msg)
                    logger.error(error_msg)
                    QMessageBox.warning(None, "警告", f"无法转换模型格式: {str(e)}")
                    plotter.close()
                    return None
            
            # 设置相机参数（默认角度，面向模型）
            plotter.camera_position = 'xz'
            plotter.camera.zoom(1.2)
            
            # 渲染场景
            plotter.render()
            
            # 捕获深度图和颜色图
            try:
                # 获取颜色缓冲区
                color_img = plotter.get_image()
                
                # 尝试使用兼容方法获取深度图
                print("[深度图修复] 尝试使用兼容方法获取深度图...")
                
                try:
                    # 方法1：尝试直接获取深度图（新版本PyVista）
                    depth_img = plotter.get_depth_image()
                    print("[深度图修复] 成功使用get_depth_image方法获取深度图")
                except (AttributeError, TypeError) as e:
                    print(f"[深度图修复] get_depth_image方法不可用，尝试VTK直接获取: {e}")
                    # 方法2：使用VTK直接获取深度缓冲区（兼容旧版本PyVista）
                    try:
                        import vtk
                        # 获取渲染窗口
                        ren_win = plotter.ren_win
                        
                        # 获取窗口大小
                        width, height = ren_win.GetSize()
                        
                        # 创建深度缓冲区
                        depth_buffer = vtk.vtkFloatArray()
                        depth_buffer.SetNumberOfComponents(1)
                        depth_buffer.SetNumberOfTuples(width * height)
                        
                        # 读取深度缓冲区
                        ren_win.GetRGBAPixelData(0, 0, width-1, height-1, 0)
                        ren_win.GetZbufferData(0, 0, width-1, height-1, depth_buffer)
                        
                        # 转换为NumPy数组并重塑
                        depth_data = np.array(depth_buffer)
                        depth_img = depth_data.reshape((height, width))
                        
                        # 反转Y轴以匹配颜色图像
                        depth_img = np.flipud(depth_img)
                        
                        print("[深度图修复] 成功使用VTK方法获取深度图")
                    except Exception as vtk_error:
                        print(f"[深度图修复] VTK方法也失败，尝试计算方法: {vtk_error}")
                        # 方法3：计算方法（作为最后备用）
                        # 获取相机信息
                        camera_pos = np.array(plotter.camera.position)
                        focal_point = np.array(plotter.camera.focal_point)
                        
                        # 计算视向
                        view_dir = focal_point - camera_pos
                        view_dir = view_dir / np.linalg.norm(view_dir)
                        
                        # 创建深度图
                        width, height = 800, 600
                        depth_img = np.zeros((height, width), dtype=np.float32)
                        
                        # 如果模型是PyVista格式，计算每个点的深度
                        if hasattr(maxilla_mesh, 'points'):
                            points = np.array(maxilla_mesh.points)
                            # 计算点到相机平面的距离
                            depths = np.dot(points - camera_pos, view_dir)
                            
                            # 规范化深度值
                            if len(depths) > 0:
                                min_depth = np.min(depths)
                                max_depth = np.max(depths)
                                if max_depth > min_depth:
                                    norm_depth = 1.0 - ((depths - min_depth) / (max_depth - min_depth))
                                    # 简单投影点到深度图（近似）
                                    for i in range(min(len(points), 5000)):  # 限制点数以提高性能
                                        # 这里使用简化的投影，实际应用中可能需要更精确的投影矩阵
                                        x = int((points[i, 0] + 100) * 2)  # 简化的坐标映射
                                        y = int((points[i, 1] + 100) * 2)  # 简化的坐标映射
                                        if 0 <= x < width and 0 <= y < height:
                                            depth_img[y, x] = norm_depth[i]
                        
                        print("[深度图修复] 成功使用计算方法生成深度图")
                
                # 处理深度图数据
                if depth_img is not None:
                    # 确保深度值在合理范围内
                    valid_depth = depth_img[depth_img > -1e9]  # 过滤无效值
                    if len(valid_depth) > 0:
                        min_depth = np.min(valid_depth)
                        max_depth = np.max(valid_depth)
                        if max_depth > min_depth:
                            depth_img = (depth_img - min_depth) / (max_depth - min_depth)
                            # 反转深度以使得近的点值大，远的点值小
                            depth_img = 1.0 - depth_img
                            # 应用伽马校正增强对比度
                            depth_img = np.power(depth_img, 0.5)
                        else:
                            depth_img = np.zeros_like(depth_img)
                    else:
                        depth_img = np.zeros_like(depth_img)
                else:
                    # 如果深度图仍然为None，创建空图像
                    depth_img = np.zeros((600, 800), dtype=np.float32)
                
                # 保存相机参数
                camera_params = {
                    'position': plotter.camera.position,
                    'focal_point': plotter.camera.focal_point,
                    'view_up': plotter.camera.view_up,
                    'distance': plotter.camera.distance,
                    'azimuth': plotter.camera.azimuth,
                    'elevation': plotter.camera.elevation
                }
                
                # 保存到投影数据中
                self.projection_data['pyvista_color_image'] = color_img
                self.projection_data['pyvista_depth_image'] = depth_img
                self.projection_data['pyvista_camera_params'] = camera_params
                
                elapsed_time = time.time() - start_time
                print(f"[深度图修复] PyVista深度图生成成功，耗时: {elapsed_time:.2f}秒")
                print(f"[深度图修复] 颜色图大小: {color_img.shape}，深度图大小: {depth_img.shape}")
                logger.info(f"PyVista深度图生成成功，大小: {color_img.shape}")
                
                # 显示成功消息
                QMessageBox.information(
                    None, 
                    "成功", 
                    "PyVista备用深度图生成成功！\n\n"+
                    f"图像大小: {color_img.shape[1]}x{color_img.shape[0]}\n"+
                    "注意：这是自动计算的深度图，可能需要在后续步骤中进行调整。"
                )
                
                plotter.close()
                return (color_img, depth_img, camera_params)
                
            except Exception as e:
                error_msg = f"[深度图修复] PyVista图像捕获错误: {e}"
                print(error_msg)
                logger.error(error_msg)
                plotter.close()
                return None
                
        except ImportError as e:
            error_msg = f"[深度图修复] PyVista导入错误: {e}"
            print(error_msg)
            logger.error(error_msg)
            QMessageBox.warning(None, "警告", "PyVista模块未正确安装，无法使用备用方法")
            return None
        except Exception as e:
            error_msg = f"[深度图修复] PyVista深度图生成异常: {e}"
            print(error_msg)
            logger.error(error_msg, exc_info=True)
            
            import traceback
            traceback_info = traceback.format_exc()
            print(f"[深度图修复] 错误详情: {traceback_info}")
            logger.debug(f"[深度图修复] 完整错误堆栈: {traceback_info}")
            
            QMessageBox.critical(None, "错误", f"PyVista深度图生成失败: {str(e)}")
            return None
    
    def generate_projection(self, grid_resolution, enable_optimization=True, optimization_level='自动', projection_type='orthographic'):
        """生成投影图像的主入口函数
        
        Args:
            grid_resolution: 网格分辨率
            enable_optimization: 是否启用优化
            optimization_level: 优化级别
            projection_type: 投影类型，'2D'、'3D'或'orthographic'(正交投影)
        """
        print("[投影生成] 开始生成投影图像...")
        
        # 添加开始时间记录
        import time
        start_time = time.time()
        
        try:
            # 检查模型是否存在，根据当前标记模式检查正确的模型
            required_model = "maxilla"  # 默认检查上颌模型
            if hasattr(self.viewer, '_marking_mode'):
                marking_mode = self.viewer._marking_mode
                if marking_mode in ["divide_mandible", "mandible_crest"]:
                    required_model = "mandible"
            
            if required_model not in self.viewer.get_models():
                QMessageBox.warning(None, "警告", f"请先加载{required_model}模型")
                return
            
            # 检查是否有拟合平面
            if not self.viewer.plane_params:
                QMessageBox.warning(None, "警告", "请先拟合平面")
                return
            
            # 检查标记点数量，考虑所有标记模式
            has_valid_points = True  # 对于平面拟合后的投影，默认允许生成
            
            # 检查当前标记模式，某些模式仍需要标记点
            if hasattr(self.viewer, '_marking_mode'):
                marking_mode = self.viewer._marking_mode
                
                # 只有特定模式需要检查标记点
                if marking_mode == "mandible_crest":
                    # 下颌后槽牙槽嵴模式，需要至少2个点
                    if hasattr(self.viewer, 'mandible_crest_points') and len(self.viewer.mandible_crest_points) >= 2:
                        has_valid_points = True
                    else:
                        has_valid_points = False
                elif marking_mode == "alveolar_ridge":
                    # 牙槽嵴模式，需要至少2个点
                    if hasattr(self.viewer, 'alveolar_ridge_points') and len(self.viewer.alveolar_ridge_points) >= 2:
                        has_valid_points = True
                    else:
                        has_valid_points = False
                elif marking_mode == "incisive_papilla":
                    # 切牙乳突模式，需要至少1个点
                    if len(self.viewer.marked_points) >= 1:
                        has_valid_points = True
                    else:
                        has_valid_points = False

            # 对于平面拟合后的投影，如果没有标记点，仍然允许生成
            if not has_valid_points:
                # 检查是否已经拟合平面，如果已经拟合，允许生成投影
                if self.viewer.plane_params:
                    has_valid_points = True
                else:
                    QMessageBox.warning(None, "警告", "根据当前标记模式，需要足够的标记点来生成投影")
                    return
                
            # 获取标记线数据，传入投影类型
            self._get_marker_lines_from_viewer(projection_type)
            
            # 根据当前标记模式选择要投影的模型
            model_type = "maxilla"  # 默认使用上颌模型
            marking_mode = getattr(self.viewer, '_marking_mode', "maxilla")
            
            if marking_mode == "divide_mandible" or marking_mode == "mandible_crest":
                model_type = "mandible"
            
            # 获取对应类型的原始模型
            model_mesh = self.viewer.get_original_model(model_type)
            if model_mesh is None:
                QMessageBox.warning(None, "警告", f"无法获取{model_type}原始模型")
                return
            print(f"[投影生成] 成功获取{model_type}原始模型")
            
            # 投影到拟合平面
            if model_type == "maxilla":
                projection_result = self.viewer.project_maxilla_to_plane(model_mesh, grid_resolution=grid_resolution)
            else:  # mandible
                # 检查是否有project_mandible_to_plane方法，如果没有则使用project_maxilla_to_plane
                if hasattr(self.viewer, 'project_mandible_to_plane'):
                    projection_result = self.viewer.project_mandible_to_plane(model_mesh, grid_resolution=grid_resolution)
                else:
                    # 如果没有专门的下颌投影方法，使用上颌投影方法
                    projection_result = self.viewer.project_maxilla_to_plane(model_mesh, grid_resolution=grid_resolution)
            
            if projection_result is None:
                QMessageBox.warning(None, "警告", "无法投影到平面")
                return
                
            projected_points_3d, triangles, depth_values = projection_result
            print(f"[投影生成] 成功投影到平面，投影点数量: {len(projected_points_3d)}")
            
            # 验证数据一致性
            if len(projected_points_3d) != len(depth_values):
                print(f"[投影生成] 数据不一致：投影点({len(projected_points_3d)})与深度值({len(depth_values)})数量不匹配")
                # 取最小值确保数据一致
                min_len = min(len(projected_points_3d), len(depth_values))
                projected_points_3d = projected_points_3d[:min_len]
                depth_values = depth_values[:min_len]
                print(f"[投影生成] 已调整数据长度为: {min_len}")
            
            # === 优化步骤 ===
            # 1. 点云预处理
            print("[投影生成] 开始点云预处理...")
            projected_points_3d, depth_values = self.preprocess_point_cloud(
                projected_points_3d, depth_values
            )
            print(f"[投影生成] 预处理后点数量: {len(projected_points_3d)}")
            
            # 移除了冗余的generate_depth_map调用，该功能已在generate_orthographic_view中实现
            print("[投影生成] 跳过冗余的深度图生成，将在后续正交投影中统一处理")
            
            # 3. 智能分辨率计算
            # 转换为2D坐标
            points_2d = self.viewer.convert_3d_to_2d(projected_points_3d)
            if points_2d is None:
                QMessageBox.warning(None, "警告", "无法转换为2D坐标")
                return
            print(f"[投影生成] 成功转换为2D坐标，2D点数量: {len(points_2d)}")
            
            optimal_resolution = self.calculate_optimal_resolution(points_2d)
            print(f"[投影生成] UI设置分辨率: {grid_resolution:.2f}mm, 计算最优分辨率: {optimal_resolution:.2f}mm")
            # 使用计算得到的最优分辨率
            grid_resolution = optimal_resolution
            
            # 保存映射关系
            self.projection_data['points_2d'] = points_2d
            self.projection_data['projected_points_3d'] = projected_points_3d
            self.projection_data['depth_values'] = depth_values
            self.projection_data['projection_type'] = 'orthographic'  # 强制使用正交投影
            
            # 投影标记线
            if self.marker_lines_3d is not None:
                self._project_marker_lines()
                if self.marker_lines_2d is not None:
                    self.projection_data['marker_lines_3d'] = self.marker_lines_3d
                    self.projection_data['marker_lines_2d'] = self.marker_lines_2d
                    print(f"[投影生成] 成功投影{len(self.marker_lines_2d)}条标记线")
                else:
                    print("[投影生成] 标记线投影失败，但继续执行深度图生成")
            
            # 4. 生成正交投影视图（只保留这一个深度图生成功能，减少资源占用）
            print("[投影生成] 开始生成正交投影视图...")
            self.generate_orthographic_view(model_mesh)
            
            # 计算总耗时
            elapsed_time = time.time() - start_time
            print(f"[投影生成] 投影过程完成，总耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            print(f"[投影生成] 生成投影时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 显示友好的错误消息
            QMessageBox.critical(
                None, 
                "投影失败", 
                f"生成投影图像时发生错误：{str(e)}\n\n请检查模型和标记点是否正确，然后重试。"
            )
            return
    
    def render_orthographic_projection_image(self, mesh, plane_params, image_size=None, margin=0.1):
        """将正交投影渲染为内存中的numpy数组
        
        Args:
            mesh: 要投影的3D模型
            plane_params: 红点标记生成的平面参数 (a, b, c, d)
            image_size: 输出图像尺寸，None表示自动计算
            margin: 图像边距比例
            
        Returns:
            numpy.ndarray: 渲染后的图像数组
        """
        print("[正交投影] 开始渲染正交投影视图...")
        
        try:
            import pyvista as pv
            import numpy as np
            
            # 创建离屏渲染器
            plotter = pv.Plotter(off_screen=True)
            
            # 设置图像尺寸
            if image_size is not None:
                plotter.window_size = image_size
            
            # 计算模型边界
            bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            max_span = max(x_span, y_span)
            
            # 添加模型到渲染器，设置半透明以确保曲线可见
            plotter.add_mesh(mesh, color='#A0C4FF', show_edges=False, opacity=0.7)
            
            # 获取当前标记模式，用于判断应该渲染哪种类型的特殊线条
            current_marking_mode = getattr(self.viewer, '_marking_mode', "maxilla")
            
            # 检查是否存在垂直平面参数，根据标记模式过滤
            if hasattr(self.viewer, 'maxilla_vertical_plane_params') and self.viewer.maxilla_vertical_plane_params:
                # 只有划分上颌模式才渲染平面交线
                if current_marking_mode == "divide_maxilla":
                    # 获取两个平面的参数
                    plane1 = plane_params  # 红点平面
                    plane2 = self.viewer.maxilla_vertical_plane_params  # 蓝点垂直平面
                    
                    # 计算两个平面的交线
                    # 平面1: a1x + b1y + c1z + d1 = 0
                    # 平面2: a2x + b2y + c2z + d2 = 0
                    a1, b1, c1, d1 = plane1
                    a2, b2, c2, d2 = plane2
                    
                    # 计算交线的方向向量：两个平面法向量的叉积
                    direction = np.cross([a1, b1, c1], [a2, b2, c2])
                    
                    # 检查两个平面是否平行
                    if np.linalg.norm(direction) < 1e-6:
                        print("[正交投影] 两个平面平行，没有交线")
                    else:
                        # 归一化方向向量
                        direction = direction / np.linalg.norm(direction)
                        
                        # 寻找交线上的一个点
                        # 选择一个变量设为0，求解另外两个变量
                        # 尝试z=0
                        if abs(a1*b2 - a2*b1) > 1e-6:
                            # 解方程组：
                            # a1x + b1y + d1 = 0
                            # a2x + b2y + d2 = 0
                            # 使用克莱姆法则
                            det = a1*b2 - a2*b1
                            det_x = -d1*b2 + d2*b1
                            det_y = -a1*d2 + a2*d1
                            x = det_x / det
                            y = det_y / det
                            z = 0
                        else:
                            # 尝试y=0
                            if abs(a1*c2 - a2*c1) > 1e-6:
                                # 解方程组：
                                # a1x + c1z + d1 = 0
                                # a2x + c2z + d2 = 0
                                det = a1*c2 - a2*c1
                                det_x = -d1*c2 + d2*c1
                                det_z = -a1*d2 + a2*d1
                                x = det_x / det
                                y = 0
                                z = det_z / det
                            else:
                                # 尝试x=0
                                det = b1*c2 - b2*c1
                                det_y = -d1*c2 + d2*c1
                                det_z = -b1*d2 + b2*d1
                                x = 0
                                y = det_y / det
                                z = det_z / det
                        
                        point = np.array([x, y, z])
                        
                        # 计算交线的长度，使用模型的对角线长度
                        line_length = np.sqrt(x_span**2 + y_span**2 + (bounds[5]-bounds[4])**2) * 2
                        
                        # 计算交线的两个端点
                        point1 = point + direction * line_length
                        point2 = point - direction * line_length
                        
                        # 创建交线的PolyData
                        line_points = np.array([point1, point2])
                        line_mesh = pv.PolyData(line_points)
                        lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
                        line_mesh.lines = lines
                        
                        # 添加交线到渲染器
                        plotter.add_mesh(
                            line_mesh,
                            color='red',
                            line_width=7.0,
                            opacity=1.0,
                            name="plane_intersection_line"
                        )
                        print("[正交投影] 添加两个平面的交线到渲染")
            
            # 不再渲染红色垂线
            # if hasattr(self.viewer, 'red_perpendicular_line_points') and self.viewer.red_perpendicular_line_points:
            #     # 只有划分上颌模式才渲染红色垂线
            #     if current_marking_mode == "divide_maxilla":
            #         # 获取红色垂线的端点
            #         line_points = np.array(self.viewer.red_perpendicular_line_points)
            #         line_mesh = pv.PolyData(line_points)
            #         lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
            #         line_mesh.lines = lines
            #         
            #         # 添加红色垂线到渲染器
            #         plotter.add_mesh(
            #             line_mesh,
            #             color='red',
            #             line_width=9.0,
            #             opacity=1.0,
            #             name="red_perpendicular_line"
            #         )
            #         print("[正交投影] 添加红色垂线到渲染")
            
            # 检查是否存在标记线数据
            if hasattr(self, 'marker_lines_3d') and self.marker_lines_3d:
                print(f"[正交投影] 开始渲染{len(self.marker_lines_3d)}条标记线")
                # 遍历所有标记线
                for i, line in enumerate(self.marker_lines_3d):
                    try:
                        # 创建标记线的PolyData
                        line_mesh = pv.PolyData(line)
                        if len(line) == 2:
                            # 只有两个点的线，创建一条简单线段
                            lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
                        else:
                            # 多个点的线（如平滑曲线），创建连续的折线
                            num_points = len(line)
                            lines = np.zeros(num_points + 1, dtype=np.int32)
                            lines[0] = num_points  # 线段包含的点数
                            lines[1:] = np.arange(num_points)  # 点的索引
                        line_mesh.lines = lines
                        
                        # 根据线类型设置不同颜色
                        color = '#00BCD4'  # 默认使用青色，优先确保maxilla线条显示（与3D视图一致）
                        line_type = "other"  # 默认类型
                        
                        # 优先使用marker_lines_info中的类型信息
                        if hasattr(self, 'marker_lines_info') and self.marker_lines_info and i < len(self.marker_lines_info):
                            line_type = self.marker_lines_info[i]
                            if line_type == "maxilla":
                                color = '#00BCD4'  # 标记上颌位置的线使用青色（与3D视图一致）
                                print(f"[正交投影] 渲染第{i}条线，类型: maxilla，颜色设置为青色")
                            else:
                                color = 'blue'  # 其他类型使用蓝色
                                print(f"[正交投影] 渲染第{i}条线，类型: {line_type}，颜色设置为蓝色")
                        elif hasattr(self.viewer, '_marking_mode') and self.viewer._marking_mode == "maxilla":
                            color = 'purple'  # 兼容旧版本，根据当前模式设置颜色
                            print(f"[正交投影] 渲染第{i}条线，根据模式设置为紫色")
                        else:
                            # 为了确保紫色线条能显示，这里做一个额外检查
                            # 如果线的点与maxilla_points中的点匹配，也设置为紫色
                            if hasattr(self.viewer, 'marked_points_modes') and hasattr(self.viewer, 'marked_points'):
                                maxilla_points = [p for i, p in enumerate(self.viewer.marked_points) if self.viewer.marked_points_modes[i] == 'maxilla']
                                if maxilla_points:
                                    maxilla_points_np = np.array(maxilla_points)
                                    for point in line:
                                        distances = np.linalg.norm(maxilla_points_np - point, axis=1)
                                        if np.any(distances < 1e-5):
                                            color = 'purple'
                                            print(f"[正交投影] 渲染第{i}条线，通过点匹配识别为maxilla，颜色设置为紫色")
                                            break
                        
                        # 添加标记线到渲染器
                        plotter.add_mesh(
                            line_mesh,
                            color=color,
                            line_width=8.0,  # 增加线宽以确保可见性
                            opacity=1.0,
                            name=f"marker_line_{i}"
                        )
                    except Exception as e:
                        print(f"[正交投影] 渲染第{i}条标记线时出错: {e}")
                        import traceback
                        traceback.print_exc()
                print(f"[正交投影] 成功添加了{len(self.marker_lines_3d)}条标记线到渲染")
            
            # 获取平面参数
            a, b, c, d = plane_params
            
            # 计算平面的法向量和中心点
            # 平面方程: ax + by + cz + d = 0
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)  # 归一化
            
            # 计算模型中心点
            center = mesh.center
            
            # 计算相机位置：确保从模型外部观察
            
            # 计算平面方程 ax + by + cz + d = 0 中的各项
            plane_eq = np.array([a, b, c, d])
            
            # 计算法向量的长度
            normal_length = np.linalg.norm(plane_eq[:3])
            
            # 创建一个足够大的包围球，确保相机在模型外部
            # 使用模型的对角线长度作为包围球直径
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            z_span = bounds[5] - bounds[4]
            model_radius = 0.5 * np.sqrt(x_span**2 + y_span**2 + z_span**2)
            
            # 为了确保相机从模型外部观察，我们需要：
            # 1. 首先确定平面法向量的方向
            # 2. 然后确保相机在模型的外部一侧
            
            # 计算模型的两个极端点（用于确定模型的范围）
            min_point = np.array([bounds[0], bounds[2], bounds[4]])
            max_point = np.array([bounds[1], bounds[3], bounds[5]])
            
            # 计算这两个点到平面的有符号距离
            min_distance = (np.dot(plane_eq[:3], min_point) + plane_eq[3]) / normal_length
            max_distance = (np.dot(plane_eq[:3], max_point) + plane_eq[3]) / normal_length
            
            # 确定模型主要位于平面的哪一侧
            # 如果最小距离为正，说明整个模型都在法向量所指的一侧
            # 如果最大距离为负，说明整个模型都在法向量的反方向一侧
            # 否则，模型穿过平面
            if min_distance > 0:
                # 模型在法向量一侧，相机应该在反方向
                normal = -normal
            elif max_distance < 0:
                # 模型在法向量反方向一侧，相机应该在法向量方向
                pass  # 保持原法向量
            else:
                # 模型穿过平面，使用模型中心到平面的距离来判断
                center_distance = (np.dot(plane_eq[:3], center) + plane_eq[3]) / normal_length
                if center_distance < 0:
                    normal = -normal
            
            dist = max_span * (1.0 + margin) * 2.0
            camera_pos = center + normal * dist  # 相机位置
            focal_point = center  # 焦点位置
            up_vector = np.array([0, 1, 0])  # 默认上方向向量
            
            # 如果法向量接近上方向向量，调整上方向
            if np.abs(np.dot(normal, up_vector)) > 0.9:
                up_vector = np.array([1, 0, 0])
            
            # 设置相机位置
            plotter.camera_position = [camera_pos, focal_point, up_vector]
            
            # 启用正交投影
            plotter.enable_parallel_projection()
            
            # 设置投影比例
            plotter.camera.parallel_scale = max_span * (0.5 + margin)
            
            # 设置背景色为白色
            plotter.set_background('white')
            
            # 执行渲染
            plotter.show(auto_close=False)
            
            # 获取渲染图像
            img = plotter.screenshot(return_img=True)
            
            # 关闭渲染器
            plotter.close()
            
            print(f"[正交投影] 渲染完成，图像尺寸: {img.shape}")
            return img
        except Exception as e:
            print(f"[正交投影] 渲染正交投影视图时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_orthographic_projection_png(self, mesh, plane_params, output_path, projection_type="maxilla", image_size=None, margin=0.1):
        """将正交投影导出为PNG文件
        
        Args:
            mesh: 要投影的3D模型
            plane_params: 红点标记生成的平面参数 (a, b, c, d)
            output_path: 输出文件路径
            projection_type: 投影类型，可以是"maxilla"、"mandible"或"alveolar"
            image_size: 输出图像尺寸，None表示自动计算
            margin: 图像边距比例
            
        Returns:
            bool: 导出成功返回True，否则返回False
        """
        print(f"[正交投影] 开始导出正交投影PNG: {output_path}...")
        
        try:
            import pyvista as pv
            import numpy as np
            
            # 获取标记线数据，传入投影类型
            self._get_marker_lines_from_viewer(projection_type)
            
            # 创建离屏渲染器
            plotter = pv.Plotter(off_screen=True)
            
            # 设置图像尺寸
            if image_size is not None:
                plotter.window_size = image_size
            
            # 添加模型到渲染器，降低透明度使曲线更突出
            plotter.add_mesh(mesh, color='#A0C4FF', show_edges=False, opacity=0.6)
            
            # 计算模型边界
            bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            max_span = max(x_span, y_span)
            
            # 检查是否存在垂直平面参数
            if hasattr(self.viewer, 'maxilla_vertical_plane_params') and self.viewer.maxilla_vertical_plane_params:
                # 获取两个平面的参数
                plane1 = plane_params  # 红点平面
                plane2 = self.viewer.maxilla_vertical_plane_params  # 蓝点垂直平面
                
                # 计算两个平面的交线
                # 平面1: a1x + b1y + c1z + d1 = 0
                # 平面2: a2x + b2y + c2z + d2 = 0
                a1, b1, c1, d1 = plane1
                a2, b2, c2, d2 = plane2
                
                # 计算交线的方向向量：两个平面法向量的叉积
                direction = np.cross([a1, b1, c1], [a2, b2, c2])
                
                # 检查两个平面是否平行
                if np.linalg.norm(direction) < 1e-6:
                    print("[正交投影] 两个平面平行，没有交线")
                else:
                    # 归一化方向向量
                    direction = direction / np.linalg.norm(direction)
                    
                    # 寻找交线上的一个点
                    # 选择一个变量设为0，求解另外两个变量
                    # 尝试z=0
                    if abs(a1*b2 - a2*b1) > 1e-6:
                        # 解方程组：
                        # a1x + b1y + d1 = 0
                        # a2x + b2y + d2 = 0
                        # 使用克莱姆法则
                        det = a1*b2 - a2*b1
                        det_x = -d1*b2 + d2*b1
                        det_y = -a1*d2 + a2*d1
                        x = det_x / det
                        y = det_y / det
                        z = 0
                    else:
                        # 尝试y=0
                        if abs(a1*c2 - a2*c1) > 1e-6:
                            # 解方程组：
                            # a1x + c1z + d1 = 0
                            # a2x + c2z + d2 = 0
                            det = a1*c2 - a2*c1
                            det_x = -d1*c2 + d2*c1
                            det_z = -a1*d2 + a2*d1
                            x = det_x / det
                            y = 0
                            z = det_z / det
                        else:
                            # 尝试x=0
                            det = b1*c2 - b2*c1
                            det_y = -d1*c2 + d2*c1
                            det_z = -b1*d2 + b2*d1
                            x = 0
                            y = det_y / det
                            z = det_z / det
                    
                    point = np.array([x, y, z])
                    
                    # 计算交线的长度，使用模型的对角线长度
                    line_length = np.sqrt(x_span**2 + y_span**2 + (bounds[5]-bounds[4])**2) * 2
                    
                    # 计算交线的两个端点
                    point1 = point + direction * line_length
                    point2 = point - direction * line_length
                    
                    # 创建交线的PolyData
                    line_points = np.array([point1, point2])
                    line_mesh = pv.PolyData(line_points)
                    lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
                    line_mesh.lines = lines
                    
                    # 添加交线到渲染器
                    plotter.add_mesh(
                        line_mesh,
                        color='red',
                        line_width=7.0,
                        opacity=1.0,
                        name="plane_intersection_line"
                    )
                    print("[正交投影] 添加两个平面的交线到渲染")
            
            # 检查是否存在红色垂线，只在生成上颌投影时渲染
            if hasattr(self.viewer, 'red_perpendicular_line_points') and self.viewer.red_perpendicular_line_points:
                # 获取当前投影类型
                current_projection_type = getattr(self, 'projection_type', None)
                # 只有在上颌投影时才渲染红色垂线
                if current_projection_type != "mandible":
                    # 获取红色垂线的端点
                    line_points = np.array(self.viewer.red_perpendicular_line_points)
                    line_mesh = pv.PolyData(line_points)
                    lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
                    line_mesh.lines = lines
                    
                    # 添加红色垂线到渲染器
                    plotter.add_mesh(
                        line_mesh,
                        color='red',
                        line_width=9.0,
                        opacity=1.0,
                        name="red_perpendicular_line"
                    )
                    print("[正交投影] 添加红色垂线到渲染")
            
            # 检查是否存在标记线数据
            if hasattr(self, 'marker_lines_3d') and self.marker_lines_3d is not None:
                # 获取当前标记模式
                current_marking_mode = getattr(self.viewer, '_marking_mode', "maxilla")
                print(f"[正交投影] 当前标记模式: {current_marking_mode}")
                print(f"[正交投影] 传入的投影类型: {projection_type}")
                
                # 根据投影类型确定应该渲染的线类型
                # 如果是上颌投影，渲染maxilla和alveolar类型的线
                # 如果是下颌投影，渲染mandible和alveolar类型的线（包括下颌牙槽嵴曲线）
                # 如果是牙槽嵴投影，渲染alveolar类型的线
                # 特别注意：对于下颌投影，我们应该渲染所有mandible类型的线，无论当前标记模式是什么
                should_render_maxilla = projection_type == "maxilla"
                should_render_mandible = projection_type == "mandible"
                should_render_alveolar = projection_type in ["alveolar", "maxilla", "mandible"]
                
                # 遍历所有标记线
                rendered_lines_count = 0
                for i, line in enumerate(self.marker_lines_3d):
                    # 获取线类型
                    line_type = "other"
                    if hasattr(self, 'marker_lines_info') and self.marker_lines_info is not None and i < len(self.marker_lines_info):
                        line_type = self.marker_lines_info[i]
                    
                    # 根据投影类型过滤标记线
                    # 只渲染与当前投影类型匹配的线
                    should_render = False
                    if should_render_maxilla and line_type == "maxilla":
                        should_render = True
                    elif should_render_mandible and line_type == "mandible":
                        should_render = True
                    elif should_render_alveolar and line_type == "alveolar":
                        should_render = True
                    
                    if not should_render:
                        print(f"[正交投影] 跳过不匹配的线，线类型: {line_type}")
                        continue
                    
                    # 创建标记线的PolyData
                    line_mesh = pv.PolyData(line)
                    if len(line) == 2:
                        # 只有两个点的线，创建一条简单线段
                        lines = np.array([2, 0, 1])  # [线段长度, 点0索引, 点1索引]
                    else:
                        # 多个点的线（如平滑曲线），创建连续的折线
                        num_points = len(line)
                        line_segments = []
                        for j in range(num_points - 1):
                            line_segments.extend([2, j, j + 1])
                        lines = np.array(line_segments)
                    line_mesh.lines = lines
                    
                    # 根据线类型设置不同颜色和属性
                    color = 'blue'  # 默认蓝色
                    line_width = 12.0  # 大幅增加线宽以确保清晰可见
                    # 根据线类型特殊处理
                    if line_type == "maxilla":
                        color = '#00BCD4'  # 标记上颌位置的线使用青色（与3D视图一致）
                        print(f"[正交投影] 渲染maxilla线，颜色设置为青色")
                    elif line_type == "mandible":
                        color = 'red'  # 标记下颌位置的线使用红色
                        print(f"[正交投影] 渲染mandible线，颜色设置为红色")
                    elif line_type == "alveolar":
                        color = 'green'  # 牙槽嵴曲线使用绿色
                        print(f"[正交投影] 渲染牙槽嵴曲线，颜色设置为绿色，线宽{line_width}")
                    
                    # 添加标记线到渲染器，确保不透明度为1.0，显示在最前面
                    # 使用render_lines_as_tubes=True确保线在任何角度都清晰可见
                    plotter.add_mesh(
                        line_mesh,
                        color=color,
                        line_width=line_width,
                        opacity=1.0,
                        render_points_as_spheres=False,
                        render_lines_as_tubes=True,  # 使用管状渲染，确保线宽一致
                        name=f"marker_line_{i}"
                    )
                    rendered_lines_count += 1
                print(f"[正交投影] 添加了{rendered_lines_count}条标记线到渲染，共{len(self.marker_lines_3d)}条可用")
            

            
            # 获取平面参数
            a, b, c, d = plane_params
            
            # 计算平面的法向量和中心点
            # 平面方程: ax + by + cz + d = 0
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)  # 归一化
            
            # 计算模型中心点
            center = mesh.center
            
            # 计算相机位置：确保从模型外部观察
            
            # 计算平面方程 ax + by + cz + d = 0 中的各项
            plane_eq = np.array([a, b, c, d])
            
            # 计算法向量的长度
            normal_length = np.linalg.norm(plane_eq[:3])
            
            # 创建一个足够大的包围球，确保相机在模型外部
            # 使用模型的对角线长度作为包围球直径
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            z_span = bounds[5] - bounds[4]
            model_radius = 0.5 * np.sqrt(x_span**2 + y_span**2 + z_span**2)
            
            # 为了确保相机从模型外部观察，我们需要：
            # 1. 首先确定平面法向量的方向
            # 2. 然后确保相机在模型的外部一侧
            
            # 计算模型的两个极端点（用于确定模型的范围）
            min_point = np.array([bounds[0], bounds[2], bounds[4]])
            max_point = np.array([bounds[1], bounds[3], bounds[5]])
            
            # 计算这两个点到平面的有符号距离
            min_distance = (np.dot(plane_eq[:3], min_point) + plane_eq[3]) / normal_length
            max_distance = (np.dot(plane_eq[:3], max_point) + plane_eq[3]) / normal_length
            
            # 确定模型主要位于平面的哪一侧
            # 如果最小距离为正，说明整个模型都在法向量所指的一侧
            # 如果最大距离为负，说明整个模型都在法向量的反方向一侧
            # 否则，模型穿过平面
            if min_distance > 0:
                # 模型在法向量一侧，相机应该在反方向
                normal = -normal
            elif max_distance < 0:
                # 模型在法向量反方向一侧，相机应该在法向量方向
                pass  # 保持原法向量
            else:
                # 模型穿过平面，使用模型中心到平面的距离来判断
                center_distance = (np.dot(plane_eq[:3], center) + plane_eq[3]) / normal_length
                if center_distance < 0:
                    normal = -normal
            
            dist = max_span * (1.0 + margin) * 2.0
            camera_pos = center + normal * dist  # 相机位置
            focal_point = center  # 焦点位置
            up_vector = np.array([0, 1, 0])  # 默认上方向向量
            
            # 如果法向量接近上方向向量，调整上方向
            if np.abs(np.dot(normal, up_vector)) > 0.9:
                up_vector = np.array([1, 0, 0])
            
            # 设置相机位置
            plotter.camera_position = [camera_pos, focal_point, up_vector]
            
            # 启用正交投影
            plotter.enable_parallel_projection()
            
            # 设置投影比例
            plotter.camera.parallel_scale = max_span * (0.5 + margin)
            
            # 设置背景色为白色
            plotter.set_background('white')
            
            # 执行渲染并保存为PNG
            plotter.show(auto_close=False)
            plotter.screenshot(output_path)
            
            # 关闭渲染器
            plotter.close()
            
            print(f"[正交投影] 正交投影PNG已导出: {output_path}")
            return True
        except Exception as e:
            print(f"[正交投影] 导出正交投影PNG时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_orthographic_view(self, mesh):
        """生成正交投影视图（使用红点标记生成的平面）
        
        Args:
            mesh: 要投影的3D模型（上颌或下颌）
        """
        try:
            print("[正交投影] 开始生成正交投影视图...")
            
            # 检查是否有拟合平面
            if not self.viewer.plane_params:
                print("[正交投影] 警告: 请先使用红点标记生成平面")
                return
            
            # 确保output目录存在
            import os
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成时间戳，确保文件名唯一
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式：YYYYMMDD_HHMMSS_mmm
            
            # 获取当前标记模式
            marking_mode = getattr(self.viewer, '_marking_mode', "maxilla")
            print(f"[正交投影] 当前标记模式: {marking_mode}")
            
            # 根据标记模式确定投影类型
            self.projection_type = "maxilla"  # 默认上颌
            if marking_mode in ["divide_mandible", "mandible_crest"]:
                self.projection_type = "mandible"
            
            # 为了保持向后兼容性，也设置局部变量
            projection_type = self.projection_type
            
            # 导出正交投影PNG，使用时间戳和投影类型生成唯一文件名
            output_path = os.path.join(output_dir, f'orthographic_projection_{projection_type}_{timestamp}.png')
            
            # 确保在导出投影前，标记模式已经正确设置
            # 保存原始标记模式
            original_marking_mode = marking_mode
            
            # 确保投影前标记模式正确
            # 对于下颌投影，确保标记模式设置为mandible_crest
            if projection_type == "mandible" and marking_mode != "mandible_crest":
                # 临时设置标记模式为mandible_crest，确保能获取到正确的投影曲线
                print(f"[正交投影] 临时将标记模式设置为mandible_crest")
                self.viewer._marking_mode = "mandible_crest"
            # 对于上颌投影，确保标记模式设置为divide_maxilla
            elif projection_type == "maxilla" and marking_mode != "divide_maxilla":
                # 临时设置标记模式为divide_maxilla，确保能获取到正确的投影曲线
                print(f"[正交投影] 临时将标记模式设置为divide_maxilla")
                self.viewer._marking_mode = "divide_maxilla"
            
            success = self.export_orthographic_projection_png(
                mesh=mesh,
                plane_params=self.viewer.plane_params,
                output_path=output_path,
                projection_type=projection_type,
                image_size=(1920, 1080),
                margin=0.1
            )
            
            # 恢复原始标记模式
            if self.viewer._marking_mode != original_marking_mode:
                print(f"[正交投影] 恢复原始标记模式: {original_marking_mode}")
                self.viewer._marking_mode = original_marking_mode
            
            if success:
                # 显示生成的图像
                try:
                    import cv2
                    print("[正交投影] 显示正交投影视图...")
                    img = cv2.imread(output_path)
                    cv2.imshow('Orthographic Projection', img)
                    cv2.waitKey(500)  # 显示0.5秒
                    cv2.destroyAllWindows()
                except Exception as cv2_err:
                    print(f"[正交投影] 显示图像失败: {cv2_err}")
            
        except Exception as e:
            print(f"[正交投影] 生成正交投影视图时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 释放资源
            import gc
            gc.collect()
            print("[正交投影] 资源已释放")
    
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
                print("[警告] 离群点过滤后点云为空，返回原始数据")
                return original_points, original_depth
            
            # 体素下采样保持均匀密度 
            downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.1) 
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # 检查下采样后是否有点
            if len(downsampled_points) == 0:
                print("[警告] 体素下采样后点云为空，返回过滤后数据")
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
            print("[警告] Open3D不可用，使用简单过滤") 
            from scipy import stats 
            z_scores = np.abs(stats.zscore(depth_values)) 
            valid_mask = z_scores < 3 
            
            filtered_points = points_3d[valid_mask]
            filtered_depth = depth_values[valid_mask]
            
            # 检查过滤后是否有点
            if len(filtered_points) == 0:
                print("[警告] 统计过滤后点云为空，返回原始数据")
                return original_points, original_depth
            
            return filtered_points, filtered_depth
    
    def calculate_optimal_resolution(self, points_2d): 
        """根据点云密度自动计算最优网格分辨率"""
        # 确保points_2d不为空
        if points_2d is None or len(points_2d) == 0:
            print("[警告] 点云为空，使用默认分辨率")
            return 0.2
        
        # 计算点云边界 
        x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
        y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 检查边界是否有效
        if x_range < 1e-6 or y_range < 1e-6:
            print("[警告] 点云范围过小，使用默认分辨率")
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
        
        print(f"[优化] 点云密度: {point_density:.2f} points/mm², 推荐分辨率: {resolution:.2f}mm") 
        return resolution
    
    def generate_optimized_depth_image(self, points_2d, depth_values, grid_resolution=0.05, 
                                      optimization_level='自动'): 
         """优化的深度图生成方法"""
         try: 
             print("[优化深度图] 开始生成优化深度图...")
              
             # 检查输入数据 
             if points_2d is None or len(points_2d) == 0: 
                 QMessageBox.warning(None, "警告", "投影点集为空") 
                 return 
              
             # 数据一致性检查 
             min_len = min(len(points_2d), len(depth_values)) 
             points_2d, depth_values = points_2d[:min_len], depth_values[:min_len] 
              
             # 根据优化级别调整参数 
             if optimization_level == '高质量':
                 # 高质量设置：更严格的异常点过滤，更多的采样点，更高的插值质量
                 print("[优化深度图] 高质量优化模式：更严格的异常点过滤")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 2*z_std) & (depth_values <= z_mean + 2*z_std)  # 更严格的过滤
                 max_points = 100000  # 更多的采样点
                 interpolation_method = 'cubic'  # 更高质量的插值
             elif optimization_level == '平衡':
                 # 平衡设置：默认参数
                 print("[优化深度图] 平衡优化模式：默认参数")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
                 max_points = 50000
                 interpolation_method = 'linear'
             elif optimization_level == '快速':
                 # 快速设置：更宽松的异常点过滤，更少的采样点，更快的插值
                 print("[优化深度图] 快速优化模式：宽松的异常点过滤，更少的采样点")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 5*z_std) & (depth_values <= z_mean + 5*z_std)  # 更宽松的过滤
                 max_points = 20000  # 更少的采样点
                 interpolation_method = 'linear'  # 更快的插值
             else:  # 自动
                 # 自动设置：根据数据量和分布自动调整
                 print("[优化深度图] 自动优化模式：根据数据自动调整参数")
                 z_mean, z_std = np.mean(depth_values), np.std(depth_values)
                 valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
                 # 根据数据量自动调整采样点数量
                 if len(points_2d) > 100000:
                     max_points = 50000
                     interpolation_method = 'linear'
                 else:
                     max_points = 100000
                     interpolation_method = 'cubic'
              
             # 异常点过滤 
             points_2d = points_2d[valid_mask] 
             depth_values = depth_values[valid_mask] 
              
             # 性能优化：降采样 
             if len(points_2d) > max_points: 
                 indices = np.random.choice(len(points_2d), max_points, replace=False) 
                 points_2d = points_2d[indices] 
                 depth_values = depth_values[indices] 
             
             # 计算范围 
             x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0]) 
             y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1]) 
             
             # 初步创建网格 - 减小扩展范围
             xi = np.arange(x_min - 1, x_max + 1, grid_resolution) 
             yi = np.arange(y_min - 1, y_max + 1, grid_resolution) 
             
             # 计算理论内存占用（每个浮点数据4字节）
             grid_size = len(xi) * len(yi)
             estimated_memory_mb = (grid_size * 4) / (1024 * 1024)  # MB
             max_memory_mb = 500  # 最大允许内存占用
             
             if estimated_memory_mb > max_memory_mb:
                 print(f"[优化深度图] 警告: 预计内存占用 {estimated_memory_mb:.2f} MB 超过阈值 {max_memory_mb} MB")
                 # 自动降低分辨率
                 new_resolution = grid_resolution * 2
                 print(f"[优化深度图] 自动将分辨率降低到 {new_resolution:.1f} mm")
                 
                 # 重新生成网格 - 减小扩展范围
                 xi = np.arange(x_min - 1, x_max + 1, new_resolution)
                 yi = np.arange(y_min - 1, y_max + 1, new_resolution)
                 grid_resolution = new_resolution
             
             # 真正创建网格
             xi, yi = np.meshgrid(xi, yi)
             print(f"[优化深度图] 平面离散化完成，网格大小: {xi.shape[0]}x{xi.shape[1]}")
             
             # 使用高级插值 
             zi = self.advanced_interpolation(points_2d, depth_values, xi, yi)
             
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
             print(f"[优化深度图] 深度值统计: {depth_stats}")
             
             # 创建图形
             fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
             print(f"[优化深度图] 创建图形完成，画布大小: 12x10, DPI: 200")
             
             # 绘制深度图
             enhanced_min = np.min(zi)
             enhanced_max = np.max(zi)
             im = ax.imshow(zi, cmap='gray', origin='lower', 
                          extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                          vmin=enhanced_min, vmax=enhanced_max, aspect='equal',
                          interpolation='lanczos')  # 使用更平滑的lanczos插值
             
             # 保存插值坐标用于联动
             self.projection_data['interpolation_coords'] = (xi, yi)
             self.projection_data['depth_image'] = zi  # 存储numpy数组而不是matplotlib图像对象
             self.projection_data['depth_ax'] = ax
             self.projection_data['depth_min'] = np.min(depth_values)
             self.projection_data['depth_max'] = np.max(depth_values)
             self.projection_data['extent'] = [x_min - 1, x_max + 1, y_min - 1, y_max + 1]
             
             # 添加颜色条，但不显示标签和刻度
             cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
             cbar.set_label('')
             cbar.set_ticks([])
              
             print("[优化深度图] 深度图绘制完成")
              
             # 保存图像和数据文件
             self._save_projection_files(fig, ax, zi)
             
             # 关闭图形
             plt.close(fig)
              
         except Exception as e:
             print(f"[优化深度图] 优化深度图生成失败: {e}")
             import traceback
             traceback.print_exc()
             # 关闭图形
             if 'fig' in locals():
                 plt.close(fig)
             # 回退到原有方法
             self.generate_2d_depth_image(points_2d, depth_values, grid_resolution)
    
    def generate_3d_depth_image(self, projected_points_3d, triangles, depth_values):
        """生成3D深度图像
        
        Args:
            projected_points_3d: 投影后的3D点坐标
            triangles: 三角面片数据
            depth_values: 每个点的深度值
        """
        print("[3D深度图] 开始生成3D深度图像...")
        
        try:
            # 检查输入数据
            if projected_points_3d is None or len(projected_points_3d) == 0:
                print("[3D深度图] 投影点集为空，无法生成图像")
                QMessageBox.warning(None, "警告", "投影点集为空，无法生成图像")
                return
            
            # 确保数据一致性
            min_len = min(len(projected_points_3d), len(depth_values))
            projected_points_3d = projected_points_3d[:min_len]
            depth_values = depth_values[:min_len]
            print(f"[3D深度图] 匹配后点数量: {len(projected_points_3d)}, 深度值数量: {len(depth_values)}")
            
            # 使用Open3D创建3D深度图像
            import open3d as o3d
            import numpy as np
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(projected_points_3d)
            
            # 将深度值作为颜色
            # 归一化深度值到[0, 1]范围
            normalized_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))
            # 创建灰度颜色
            colors = np.zeros((len(normalized_depth), 3))
            colors[:, 0] = normalized_depth  # R
            colors[:, 1] = normalized_depth  # G
            colors[:, 2] = normalized_depth  # B
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 如果有三角面片数据，创建网格对象
            if triangles is not None and len(triangles) > 0:
                # 确保三角面片索引在有效范围内
                valid_triangles = []
                for tri in triangles:
                    if all(index < len(projected_points_3d) for index in tri):
                        valid_triangles.append(tri)
                
                if valid_triangles:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(projected_points_3d)
                    mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
                    # 计算法线
                    mesh.compute_vertex_normals()
                    # 设置颜色
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                    
                    # 保存网格
                    o3d.io.write_triangle_mesh("projection_3d_mesh.ply", mesh)
                    print("[3D深度图] 已保存3D网格模型: projection_3d_mesh.ply")
            
            # 保存点云
            o3d.io.write_point_cloud("projection_3d_pointcloud.ply", pcd)
            print("[3D深度图] 已保存3D点云: projection_3d_pointcloud.ply")
            
            # 可视化3D深度图
            print("[3D深度图] 可视化3D深度图...")
            o3d.visualization.draw_geometries([pcd])
            
            # 保存数据到投影数据字典
            self.projection_data['depth_image_3d'] = pcd
            self.projection_data['depth_min'] = np.min(depth_values)
            self.projection_data['depth_max'] = np.max(depth_values)
            
            print("[3D深度图] 3D深度图生成完成")
            
        except Exception as e:
            print(f"[3D深度图] 生成3D深度图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_2d_depth_image(self, points_2d, depth_values, grid_resolution=0.05, 
                              interpolation_method='linear'):
        """生成2D深度图的原始方法"""
        print("[原始深度图] 开始生成2D深度图...")
        
        try:
            # 检查输入数据
            if points_2d is None or len(points_2d) == 0:
                print("[原始深度图] 投影点集为空，无法生成图像")
                QMessageBox.warning(None, "警告", "投影点集为空，无法生成图像")
                return
            
            # 确保数据一致性
            min_len = min(len(points_2d), len(depth_values))
            points_2d, depth_values = points_2d[:min_len], depth_values[:min_len]
            print(f"[原始深度图] 匹配后点数量: {len(points_2d)}, 深度值数量: {len(depth_values)}")
            
            # 异常点处理：过滤超出均值±3倍标准差的深度值
            print("[原始深度图] 开始异常点处理...")
            z_mean, z_std = np.mean(depth_values), np.std(depth_values)
            valid_mask = (depth_values >= z_mean - 3*z_std) & (depth_values <= z_mean + 3*z_std)
            filtered_points_2d = points_2d[valid_mask]
            filtered_depth_values = depth_values[valid_mask]
            
            filtered_count = len(points_2d) - len(filtered_points_2d)
            if filtered_count > 0:
                print(f"[原始深度图] 已过滤 {filtered_count} 个离群点")
                points_2d = filtered_points_2d
                depth_values = filtered_depth_values
            else:
                print("[原始深度图] 未检测到离群点")
            
            # 优化：仅在点数量过多时进行降采样
            if len(points_2d) > 50000:
                print(f"[原始深度图] 原始点数量: {len(points_2d)}")
                indices = np.random.choice(len(points_2d), 50000, replace=False)
                points_2d = points_2d[indices]
                depth_values = depth_values[indices]
                print(f"[原始深度图] 降采样后点数量: {len(points_2d)}")
            
            # 计算投影点范围
            x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
            y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
            print(f"[原始深度图] 投影点集范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
            print(f"[原始深度图] 创建图形完成，画布大小: 12x10, DPI: 200")
            
            # 创建网格 - 减小扩展范围
            xi = np.arange(x_min - 1, x_max + 1, grid_resolution)
            yi = np.arange(y_min - 1, y_max + 1, grid_resolution)
            xi, yi = np.meshgrid(xi, yi)
            print(f"[原始深度图] 平面离散化完成，网格大小: {xi.shape[0]}x{xi.shape[1]}")
            
            # 插值生成深度图
            print(f"[原始深度图] 开始生成深度图，使用{interpolation_method}插值...")
            try:
                # 使用高级插值算法生成深度图
                zi = self.advanced_interpolation(points_2d, depth_values, xi, yi)
            except Exception as e:
                print(f"[原始深度图] 高级插值失败，回退到标准插值: {e}")
                # 使用标准插值方法作为备选
                try:
                    from scipy.interpolate import griddata
                    # 使用指定插值方法生成深度图
                    zi = griddata(points_2d, depth_values, (xi, yi), method=interpolation_method)
                    
                    # 处理插值失败的点
                    if np.any(np.isnan(zi)):
                        print("[原始深度图] 检测到NaN值，使用最近邻插值填充...")
                        nan_mask = np.isnan(zi)
                        zi[nan_mask] = griddata(points_2d, depth_values, 
                                             (xi[nan_mask], yi[nan_mask]), method='nearest')
                except Exception as e:
                    print(f"[原始深度图] 线性插值失败，回退到最近邻插值: {e}")
                    from scipy.interpolate import griddata
                    zi = griddata(points_2d, depth_values, (xi, yi), method='nearest')
            
            # 绘制深度图
            enhanced_min = np.min(zi)
            enhanced_max = np.max(zi)
            im = ax.imshow(zi, cmap='gray', origin='lower', 
                         extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                         vmin=enhanced_min, vmax=enhanced_max, aspect='equal',
                         interpolation='lanczos')
            
            # 保存插值坐标用于联动
            self.projection_data['interpolation_coords'] = (xi, yi)
            self.projection_data['depth_image'] = im
            self.projection_data['depth_ax'] = ax
            self.projection_data['depth_min'] = np.min(depth_values)
            self.projection_data['depth_max'] = np.max(depth_values)
            self.projection_data['extent'] = [x_min - 2, x_max + 2, y_min - 2, y_max + 2]
            
            print("[原始深度图] 深度图绘制完成")
            
            # 保存图像和数据文件
            self._save_projection_files(fig, ax, zi)
            
            # 关闭图形
            plt.close(fig)
            
        except Exception as e:
            print(f"[原始深度图] 生成深度图失败: {e}")
            import traceback
            traceback.print_exc()
            # 关闭图形
            if 'fig' in locals():
                plt.close(fig)
    
    def advanced_interpolation(self, points_2d, depth_values, xi, yi): 
        """增强的插值算法 - 优化版"""
        from scipy.interpolate import Rbf, griddata
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
            # 先尝试使用多二次曲面径向基函数插值处理NaN区域
            try:
                rbf = Rbf(points_2d[:, 0], points_2d[:, 1], depth_values, function='multiquadric', epsilon=2*std_depth)
                zi_rbf = rbf(xi, yi)
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
        
        # 深度图质量增强
        zi = self.enhance_depth_map_quality(zi, depth_values)
        
        return zi 
    
    def edge_preserving_smooth(self, depth_map): 
        """高级边缘保持平滑 - 优化版"""
        from scipy.ndimage import median_filter, sobel
        
        # 先使用中值滤波去除椒盐噪声
        median_filtered = median_filter(depth_map, size=3)
        
        # 计算深度图的梯度（边缘）
        depth_grad_x = sobel(median_filtered, axis=0)
        depth_grad_y = sobel(median_filtered, axis=1)
        depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        depth_grad_mag = (depth_grad_mag - np.min(depth_grad_mag)) / (np.max(depth_grad_mag) - np.min(depth_grad_mag) + 1e-10)
        
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
        
        # 1. 动态范围调整 - 使用更精确的百分位
        depth_1 = np.percentile(depth_values, 1)
        depth_99 = np.percentile(depth_values, 99)
        
        # 2. 压缩极端值，增强中间范围对比度
        zi_clipped = np.clip(zi, depth_1, depth_99)
        
        # 3. 线性拉伸到完整范围
        if depth_99 > depth_1:
            zi_stretched = (zi_clipped - depth_1) / (depth_99 - depth_1)
            
            # 4. 应用基于直方图的非线性增强
            hist, bins = np.histogram(zi_stretched, bins=256, range=(0, 1))
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
              
            # 使用累积分布函数进行直方图均衡化
            zi_enhanced = np.interp(zi_stretched.flatten(), bins[:-1], cdf_normalized)
            zi_enhanced = zi_enhanced.reshape(zi_stretched.shape)
            
            # 5. 应用轻微的伽马校正
            mean_depth = np.mean(zi_stretched)
            gamma = 1.1 if mean_depth < 0.5 else 0.9
            zi_enhanced = np.power(zi_enhanced, gamma)
        else:
            zi_enhanced = zi_clipped
            # 如果范围太小，归一化到0-1区间
            if np.max(zi_enhanced) > np.min(zi_enhanced):
                zi_enhanced = (zi_enhanced - np.min(zi_enhanced)) / (np.max(zi_enhanced) - np.min(zi_enhanced))
        
        return zi_enhanced
        
    def _get_marker_lines_from_viewer(self, projection_type=None):
        """从viewer中获取标记线数据，支持所有标记模式
        
        Args:
            projection_type: 投影类型，可以是"maxilla"、"mandible"或None
        """
        print("[投影生成] 开始获取标记线数据...")
        try:
            # 初始化标记线数据
            self.marker_lines_3d = []
            self.marker_lines_info = []  # 添加线类型信息列表
            
            # 获取当前标记模式
            marking_mode = getattr(self.viewer, '_marking_mode', "maxilla")
            print(f"[投影生成] 当前标记模式: {marking_mode}")
            print(f"[投影生成] 投影类型: {projection_type}")
            
            # 根据投影类型或标记模式获取相关的标记线
            # 如果指定了投影类型，优先根据投影类型获取标记线
            
            # 处理上颌标记线
            if projection_type == "maxilla" or marking_mode == 'divide_maxilla':
                # 划分上颌模式：使用上颌相关的平滑曲线
                # 首先检查并打印divide_maxilla_curve的状态
                has_curve = hasattr(self.viewer, 'divide_maxilla_curve')
                is_list = isinstance(self.viewer.divide_maxilla_curve, list) if has_curve else False
                curve_length = len(self.viewer.divide_maxilla_curve) if has_curve and is_list else 0
                print(f"[投影生成] divide_maxilla_curve状态: has={has_curve}, is_list={is_list}, length={curve_length}")
                
                # 优先使用平滑曲线，即使只有一个点也尝试使用（确保至少有两个点）
                if hasattr(self.viewer, 'divide_maxilla_curve') and isinstance(self.viewer.divide_maxilla_curve, (list, np.ndarray)) and len(self.viewer.divide_maxilla_curve) >= 2:
                    # 优先使用平滑曲线
                    maxilla_line = self.viewer.divide_maxilla_curve
                    if isinstance(maxilla_line, np.ndarray):
                        maxilla_line = maxilla_line.tolist()
                    self.marker_lines_3d.append(np.array(maxilla_line))
                    self.marker_lines_info.append("maxilla")
                    print(f"[投影生成] 划分上颌模式: 使用划分上颌平滑曲线，共 {len(maxilla_line)} 个点")
                elif hasattr(self.viewer, 'divide_maxilla_points') and isinstance(self.viewer.divide_maxilla_points, (list, np.ndarray)) and len(self.viewer.divide_maxilla_points) >= 2:
                    # 如果没有平滑曲线，回退到使用原始点
                    maxilla_line = self.viewer.divide_maxilla_points
                    if isinstance(maxilla_line, np.ndarray):
                        maxilla_line = maxilla_line.tolist()
                    self.marker_lines_3d.append(np.array(maxilla_line))
                    self.marker_lines_info.append("maxilla")
                    print(f"[投影生成] 划分上颌模式: 使用划分上颌原始标记线，共 {len(maxilla_line)} 个点")
                else:
                    # 尝试直接从viewer中获取其他可能的曲线数据
                    print("[投影生成] 尝试获取其他可能的曲线数据...")
                    # 检查是否有alveolar_ridge_curve（可能用户实际使用的是这个）
                    if hasattr(self.viewer, 'alveolar_ridge_curve') and isinstance(self.viewer.alveolar_ridge_curve, (list, np.ndarray)) and len(self.viewer.alveolar_ridge_curve) >= 2:
                        ridge_curve = self.viewer.alveolar_ridge_curve
                        if isinstance(ridge_curve, np.ndarray):
                            ridge_curve = ridge_curve.tolist()
                        self.marker_lines_3d.append(np.array(ridge_curve))
                        self.marker_lines_info.append("maxilla")  # 作为maxilla类型处理
                        print(f"[投影生成] 划分上颌模式: 使用牙槽嵴拟合曲线，共 {len(ridge_curve)} 个点")
                    else:
                        print("[投影生成] 未找到有效的上颌标记线数据")
            # 处理下颌标记线
            elif projection_type == "mandible" or marking_mode in ['divide_mandible', 'mandible_crest']:
                # 划分下颌或下颌牙槽嵴模式：使用下颌相关的标记线
                
                # 首先检查划分下颌标记线，无论当前标记模式是什么
                # 这确保了在任何下颌投影情况下，都能获取到划分下颌的标记线
                if hasattr(self.viewer, 'divide_mandible_points') and isinstance(self.viewer.divide_mandible_points, list) and len(self.viewer.divide_mandible_points) >= 2:
                    mandible_line = self.viewer.divide_mandible_points
                    self.marker_lines_3d.append(np.array(mandible_line))
                    self.marker_lines_info.append("mandible")
                    print(f"[投影生成] 使用划分下颌标记线，共 {len(mandible_line)} 个点")
                
                # 检查下颌牙槽嵴拟合曲线，无论当前标记模式是什么
                # 这样可以确保在生成投影时，即使标记模式切换了，也能显示下颌牙槽嵴的拟合曲线
                if hasattr(self.viewer, 'mandible_crest_curve') and isinstance(self.viewer.mandible_crest_curve, (list, np.ndarray)) and len(self.viewer.mandible_crest_curve) >= 2:
                    crest_line = self.viewer.mandible_crest_curve
                    if isinstance(crest_line, np.ndarray):
                        crest_line = crest_line.tolist()
                    self.marker_lines_3d.append(np.array(crest_line))
                    self.marker_lines_info.append("mandible")  # 牙槽嵴线也作为下颌线类型处理
                    print(f"[投影生成] 使用下颌牙槽嵴拟合曲线，共 {len(crest_line)} 个点")
                elif hasattr(self.viewer, 'mandible_crest_points') and isinstance(self.viewer.mandible_crest_points, list) and len(self.viewer.mandible_crest_points) >= 2:
                    crest_line = self.viewer.mandible_crest_points
                    self.marker_lines_3d.append(np.array(crest_line))
                    self.marker_lines_info.append("mandible")  # 牙槽嵴线也作为下颌线类型处理
                    print(f"[投影生成] 使用下颌牙槽嵴标记线，共 {len(crest_line)} 个点")
                else:
                    # 尝试直接从viewer中获取其他可能的曲线数据
                    print("[投影生成] 尝试获取其他可能的曲线数据...")
                    # 检查是否有alveolar_ridge_curve（可能用户实际使用的是这个）
                    if hasattr(self.viewer, 'alveolar_ridge_curve') and isinstance(self.viewer.alveolar_ridge_curve, (list, np.ndarray)) and len(self.viewer.alveolar_ridge_curve) >= 2:
                        ridge_curve = self.viewer.alveolar_ridge_curve
                        if isinstance(ridge_curve, np.ndarray):
                            ridge_curve = ridge_curve.tolist()
                        self.marker_lines_3d.append(np.array(ridge_curve))
                        self.marker_lines_info.append("mandible")  # 作为mandible类型处理
                        print(f"[投影生成] 下颌牙槽嵴模式: 使用牙槽嵴拟合曲线，共 {len(ridge_curve)} 个点")
                    else:
                        print("[投影生成] 未找到有效的下颌标记线数据")
            elif marking_mode == 'alveolar_ridge':
                # 牙槽嵴模式：使用牙槽嵴拟合曲线
                if hasattr(self.viewer, 'alveolar_ridge_curve') and isinstance(self.viewer.alveolar_ridge_curve, list) and len(self.viewer.alveolar_ridge_curve) >= 2:
                    ridge_curve = self.viewer.alveolar_ridge_curve
                    self.marker_lines_3d.append(np.array(ridge_curve))
                    self.marker_lines_info.append("alveolar")  # 牙槽嵴曲线专用类型
                    print(f"[投影生成] 牙槽嵴模式: 使用牙槽嵴拟合曲线，共 {len(ridge_curve)} 个点")
            
            # 检查并添加投影曲线
            if hasattr(self.viewer, 'projected_marker_lines') and isinstance(self.viewer.projected_marker_lines, list):
                for line_data in self.viewer.projected_marker_lines:
                    if 'type' in line_data and 'points' in line_data:
                        line_type = line_data['type']
                        points = line_data['points']
                        
                        # 确保投影曲线总是被添加，只要它与当前投影类型匹配
                        if isinstance(points, list) and len(points) >= 2:
                            # 添加上颌投影曲线
                            if line_type == 'alveolar_ridge_projection':
                                self.marker_lines_3d.append(np.array(points))
                                self.marker_lines_info.append("maxilla")
                                print(f"[投影生成] 添加上颌投影曲线，共 {len(points)} 个点")
                            # 添加下颌投影曲线
                            elif line_type == 'mandible_crest_projection':
                                self.marker_lines_3d.append(np.array(points))
                                self.marker_lines_info.append("mandible")
                                print(f"[投影生成] 添加下颌投影曲线，共 {len(points)} 个点")
            
            # 移除红色垂线（标记点前方8mm的线段），避免在投影中显示
            if hasattr(self.viewer, 'red_perpendicular_line_points') and self.viewer.red_perpendicular_line_points:
                red_line = np.array(self.viewer.red_perpendicular_line_points)
                # 检查是否在标记线列表中，如果存在则移除
                for i, line in enumerate(self.marker_lines_3d):
                    # 只检查两点线段（红色垂线只有两个点）
                    if line.shape == (2, 3):
                        # 使用容错比较，解决浮点数精度问题
                        if (np.allclose(line, red_line, rtol=1e-6, atol=1e-3) or 
                            np.allclose(line, np.flipud(red_line), rtol=1e-6, atol=1e-3)):
                            del self.marker_lines_3d[i]
                            del self.marker_lines_info[i]
                            break
            print("[投影生成] 已优化标记线获取逻辑，只获取与当前标记模式相关的标记线")
            
            # 4. 去重处理，确保投影曲线总是被保留
            # 首先将所有投影曲线单独提取出来
            projection_lines = []
            projection_info = []
            original_lines = []
            original_info = []
            
            # 标记哪些线是投影曲线
            for i, line in enumerate(self.marker_lines_3d):
                is_projection = False
                if hasattr(self.viewer, 'projected_marker_lines') and isinstance(self.viewer.projected_marker_lines, list):
                    for proj_line in self.viewer.projected_marker_lines:
                        if 'points' in proj_line:
                            proj_points = np.array(proj_line['points'])
                            if np.array_equal(line, proj_points) or np.array_equal(line, np.flipud(proj_points)):
                                is_projection = True
                                break
                
                if is_projection:
                    projection_lines.append(line)
                    projection_info.append(self.marker_lines_info[i])
                else:
                    original_lines.append(line)
                    original_info.append(self.marker_lines_info[i])
            
            print(f"[投影生成] 投影曲线数量: {len(projection_lines)}, 原始曲线数量: {len(original_lines)}")
            
            # 对原始曲线进行去重
            unique_original_lines = []
            unique_original_info = []
            for i, line in enumerate(original_lines):
                is_duplicate = False
                for j, unique_line in enumerate(unique_original_lines):
                    if np.array_equal(line, unique_line) or np.array_equal(line, np.flipud(unique_line)):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_original_lines.append(line)
                    unique_original_info.append(original_info[i])
            
            print(f"[投影生成] 去重后原始曲线数量: {len(unique_original_lines)}")
            
            # 合并投影曲线和去重后的原始曲线
            # 确保投影曲线总是在前面
            self.marker_lines_3d = projection_lines + unique_original_lines
            self.marker_lines_info = projection_info + unique_original_info
            
            # 5. 完整性检查和调试信息
            print(f"[投影生成] 最终标记线数量: {len(self.marker_lines_3d)}")
            if self.marker_lines_3d:
                print(f"[投影生成] 标记线类型分布: maxilla={sum(1 for info in self.marker_lines_info if info == 'maxilla')}, mandible={sum(1 for info in self.marker_lines_info if info == 'mandible')}, other={sum(1 for info in self.marker_lines_info if info not in ['maxilla', 'mandible'])}")
            
            # 确保标记线数据不为None，避免后续渲染出错
            if not self.marker_lines_3d:
                self.marker_lines_3d = []
                self.marker_lines_info = []
                print(f"[投影生成] 没有找到有效的标记线数据")
            
            # 最终状态报告
            if self.marker_lines_3d:
                print(f"[投影生成] 成功获取并处理{len(self.marker_lines_3d)}条有效标记线")
                for i, info in enumerate(self.marker_lines_info):
                    print(f"[投影生成] 标记线{i+1}类型: {info}")
            else:
                # 如果所有尝试都失败
                print("[投影生成] 未找到有效标记线数据")
                self.marker_lines_3d = None
                self.marker_lines_info = None
        except Exception as e:
            print(f"[投影生成] 获取标记线数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.marker_lines_3d = None
            self.marker_lines_info = None
        except Exception as e:
            print(f"[投影生成] 获取标记线数据时出错: {e}")
            import traceback
            print(f"[投影生成] 错误详情: {traceback.format_exc()}")
            self.marker_lines_3d = None
            self.marker_lines_info = None
    
    def _project_marker_lines(self):
        """将3D标记线投影到2D平面"""
        print("[投影生成] 开始投影标记线...")
        try:
            if self.marker_lines_3d is None:
                print("[投影生成] 没有3D标记线数据，跳过投影")
                self.marker_lines_2d = None
                return
            
            # 使用viewer的convert_3d_to_2d方法将每条线段的端点投影到2D
            self.marker_lines_2d = []
            valid_lines_count = 0
            
            for line_idx, line in enumerate(self.marker_lines_3d):
                try:
                    # 确保line是numpy数组
                    line_3d = np.array(line)
                    
                    # 检查数据有效性
                    if line_3d.ndim != 2 or line_3d.shape[1] != 3:
                        print(f"[投影生成] 跳过无效的3D线段格式: 索引{line_idx}")
                        continue
                    
                    # 投影每个点
                    line_2d = []
                    all_points_valid = True
                    
                    for point_3d in line_3d:
                        try:
                            # 使用viewer的3D到2D转换方法
                            point_2d_result = self.viewer.convert_3d_to_2d([point_3d])
                            if point_2d_result is not None and len(point_2d_result) > 0:
                                point_2d = point_2d_result[0]
                                # 检查2D点是否有效
                                if isinstance(point_2d, (list, np.ndarray)) and len(point_2d) >= 2:
                                    line_2d.append(point_2d)
                                else:
                                    print(f"[投影生成] 无效的2D点: {point_2d}")
                                    all_points_valid = False
                                    break
                            else:
                                print(f"[投影生成] 3D到2D转换失败")
                                all_points_valid = False
                                break
                        except Exception as point_error:
                            print(f"[投影生成] 投影单个点时出错: {point_error}")
                            all_points_valid = False
                            break
                    
                    if all_points_valid and len(line_2d) >= 2:
                        self.marker_lines_2d.append(np.array(line_2d))
                        valid_lines_count += 1
                    else:
                        print(f"[投影生成] 线段{line_idx}包含无效点，跳过")
                        
                except Exception as line_error:
                    print(f"[投影生成] 处理线段{line_idx}时出错: {line_error}")
                    continue
                
            # 只有在有有效线段时才转换为numpy数组
            if self.marker_lines_2d:
                self.marker_lines_2d = np.array(self.marker_lines_2d)
                print(f"[投影生成] 成功将{valid_lines_count}条3D标记线投影到2D（总处理{len(self.marker_lines_3d)}条）")
            else:
                print("[投影生成] 没有成功投影的标记线")
                self.marker_lines_2d = None
                
        except Exception as e:
            print(f"[投影生成] 投影标记线时出错: {e}")
            import traceback
            print(f"[投影生成] 错误详情: {traceback.format_exc()}")
            self.marker_lines_2d = None
    
    def _save_projection_files(self, fig, ax, zi):
        """保存投影图像和相关数据文件"""
        try:
            # 绘制标记线
            if self.marker_lines_2d is not None and len(self.marker_lines_2d) > 0:
                print("[文件保存] 绘制标记线到深度图...")
                for line in self.marker_lines_2d:
                    ax.plot(line[:, 0], line[:, 1], color='red', linewidth=2, alpha=0.8)
                    # 为每个线段添加端点标签
                    for i, point in enumerate(line):
                        ax.text(point[0], point[1], f'{i+1}', color='white', fontsize=8, 
                                ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7, pad=2))
            
            # 设置输出目录为output文件夹
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            # 确保output目录存在
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成时间戳，确保文件名唯一
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式：YYYYMMDD_HHMMSS_mmm
            
            # 保存深度图为高DPI的PNG
            output_path_depth = os.path.join(output_dir, f"depth_map_{timestamp}.png")
            print(f"[文件保存] 准备保存深度图到: {output_path_depth}")
            plt.savefig(output_path_depth, dpi=500, bbox_inches='tight', pad_inches=0.1, format='png', 
                      pil_kwargs={'quality': 95, 'optimize': True}, antialiased=True, rasterized=True)
            print(f"[文件保存] 深度图保存完成: {output_path_depth}")
            
            # 保存投影图像为SVG
            output_path_svg = os.path.join(output_dir, f"projection_{timestamp}.svg")
            print(f"[文件保存] 准备保存投影图像到: {output_path_svg}")
            plt.savefig(output_path_svg, format='svg', bbox_inches='tight', pad_inches=0.1)
            print(f"[文件保存] SVG图像保存完成: {output_path_svg}")
            
            # 保存深度矩阵为npy文件（用于后续分析）
            output_path_depth_matrix = os.path.join(output_dir, f"depth_matrix_{timestamp}.npy")
            print(f"[文件保存] 准备保存深度矩阵到: {output_path_depth_matrix}")
            np.save(output_path_depth_matrix, zi)
            print(f"[文件保存] 深度矩阵保存完成: {output_path_depth_matrix}")
            
            # 保存深度值为CSV格式（便于通用软件分析）
            output_path_csv = os.path.join(output_dir, f"depth_values_{timestamp}.csv")
            print(f"[文件保存] 准备保存深度值CSV到: {output_path_csv}")
            # 创建包含2D坐标和对应深度值的结构化数据
            csv_data = np.column_stack((self.projection_data['points_2d'], self.projection_data['depth_values']))
            header = "x,y,depth"
            np.savetxt(output_path_csv, csv_data, delimiter=",", header=header, comments="")
            print(f"[文件保存] 深度值CSV保存完成: {output_path_csv}")
            
            # 保存投影点云为PLY格式（便于3D软件查看）
            output_path_ply = os.path.join(output_dir, f"projected_points_{timestamp}.ply")
            print(f"[文件保存] 准备保存投影点云到: {output_path_ply}")
            # 创建Open3D点云对象
            import open3d as o3d
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
            print(f"[文件保存] 投影点云保存完成: {output_path_ply}")
            
            # 保存标记线数据
            if self.marker_lines_2d is not None and len(self.marker_lines_2d) > 0:
                output_path_markers = os.path.join(output_dir, f"marker_lines_{timestamp}.csv")
                print(f"[文件保存] 准备保存标记线数据到: {output_path_markers}")
                try:
                    # 创建标记线CSV数据
                    marker_data = []
                    for line_idx, line in enumerate(self.marker_lines_2d):
                        for point_idx, point in enumerate(line):
                            marker_data.append([line_idx + 1, point_idx + 1, point[0], point[1]])
                    
                    # 保存标记线CSV
                    header = "line_id,point_id,x,y"
                    np.savetxt(output_path_markers, marker_data, delimiter=",", header=header, comments="")
                    print(f"[文件保存] 标记线数据保存完成: {output_path_markers}")
                    
                except Exception as e:
                    print(f"[文件保存] 保存标记线数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[文件保存] 保存文件时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 关闭图形，释放内存
            plt.close(fig)
                
    def get_projection_data(self):
        """获取投影数据"""
        return self.projection_data
    
    def clear_projection_data(self):
        """清除投影数据"""
        self.projection_data = {
            'points_2d': None,
            'projected_points_3d': None,
            'depth_values': None,
            'interpolation_coords': None,
            'depth_image': None,
            'depth_ax': None,
            'depth_min': None,
            'depth_max': None,
            'extent': None,
            'marker_lines_3d': None,
            'marker_lines_2d': None
        }
        self.marker_lines_3d = None
        self.marker_lines_2d = None
    
    def get_depth_image(self):
        """获取深度图像"""
        return self.projection_data['depth_image']
    
    def get_marker_lines_2d(self):
        """获取2D标记线"""
        return self.marker_lines_2d
    
    def get_marker_lines_3d(self):
        """获取3D标记线"""
        return self.marker_lines_3d
    
    def set_marker_lines_3d(self, marker_lines_3d):
        """设置3D标记线"""
        self.marker_lines_3d = marker_lines_3d
    
    def update_projection_data(self, new_data):
        """更新投影数据"""
        self.projection_data.update(new_data)
    
    def get_depth_min(self):
        """获取最小深度值"""
        return self.projection_data['depth_min']
    
    def get_depth_max(self):
        """获取最大深度值"""
        return self.projection_data['depth_max']
    
    def get_extent(self):
        """获取深度图范围"""
        return self.projection_data['extent']
    
    def get_interpolation_coords(self):
        """获取插值坐标"""
        return self.projection_data['interpolation_coords']
    
    def get_depth_ax(self):
        """获取深度图坐标轴"""
        return self.projection_data['depth_ax']
    
    def get_depth_ax(self):
        """获取深度图坐标轴"""
        return self.projection_data['depth_ax']
    
    def set_depth_ax(self, ax):
        """设置深度图坐标轴"""
        self.projection_data['depth_ax'] = ax
    
    def get_points_2d(self):
        """获取2D点"""
        return self.projection_data['points_2d']
    
    def get_projected_points_3d(self):
        """获取投影的3D点"""
        return self.projection_data['projected_points_3d']
    
    def get_depth_values(self):
        """获取深度值"""
        return self.projection_data['depth_values']
    
    def get_depth_ax(self):
        """获取深度图坐标轴"""
        return self.projection_data['depth_ax']
    
    def set_depth_ax(self, ax):
        """设置深度图坐标轴"""
        self.projection_data['depth_ax'] = ax
    
    def get_points_2d(self):
        """获取2D点"""
        return self.projection_data['points_2d']