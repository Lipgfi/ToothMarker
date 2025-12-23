import os
import numpy as np
import open3d as o3d
import pyvista as pv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """模型加载器，用于加载STL模型文件"""
    
    def __init__(self):
        pass
    
    def load_stl(self, file_path):
        """加载STL文件并返回Open3D的TriangleMesh对象
        
        Args:
            file_path: STL文件路径
            
        Returns:
            open3d.geometry.TriangleMesh: 加载的模型
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None
            
            # 直接使用PyVista加载STL文件，它能更好地处理二进制格式
            logger.info(f"开始加载STL文件: {file_path}")
            
            # 使用PyVista的read函数，它能自动识别并处理二进制STL
            pv_mesh = pv.read(file_path)
            
            logger.info(f"PyVista成功加载STL文件")
            logger.info(f"顶点数: {pv_mesh.n_points}")
            logger.info(f"三角形数: {pv_mesh.n_cells}")
            
            # 检查是否成功加载到数据
            if pv_mesh.n_points == 0 or pv_mesh.n_cells == 0:
                logger.error(f"加载的模型数据为空: {file_path}")
                return None
            
            # 从PyVista网格提取顶点和面数据
            vertices = np.array(pv_mesh.points, dtype=np.float64)
            
            # 处理PyVista的面数据格式
            faces_data = np.array(pv_mesh.faces)
            # PyVista的face格式是: [n_points, p1, p2, p3, n_points, p1, p2, p3, ...]
            triangles = []
            i = 0
            while i < len(faces_data):
                n_points = int(faces_data[i])
                if n_points == 3:  # 三角形
                    triangles.append([int(faces_data[i+1]), int(faces_data[i+2]), int(faces_data[i+3])])
                i += n_points + 1
            
            triangles = np.array(triangles, dtype=np.int32)
            
            logger.info(f"成功提取{len(triangles)}个三角形")
            
            # 创建Open3D网格对象
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # 计算法向量
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            logger.info(f"成功创建Open3D网格并计算法向量")
            
            return mesh
            
        except Exception as e:
            logger.error(f"加载STL文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None