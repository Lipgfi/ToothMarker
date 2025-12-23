import os
import numpy as np
import open3d as o3d
import pyvista as pv
import time
from datetime import datetime


def save_mesh(mesh, file_path, file_format='stl'):
    """保存网格到文件
    
    Args:
        mesh: Open3D网格对象
        file_path: 文件路径
        file_format: 文件格式（'stl', 'ply', 'obj'）
        
    Returns:
        是否保存成功
    """
    print(f"保存网格到文件: {file_path}")
    
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        if file_format.lower() == 'stl':
            success = o3d.io.write_triangle_mesh(file_path, mesh, write_triangle_uvs=False)
        elif file_format.lower() == 'ply':
            success = o3d.io.write_triangle_mesh(file_path, mesh, write_ascii=True)
        elif file_format.lower() == 'obj':
            success = o3d.io.write_triangle_mesh(file_path, mesh, write_triangle_uvs=True)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        if success:
            print(f"网格保存成功，文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
            return True
        else:
            print("网格保存失败")
            return False
            
    except Exception as e:
        print(f"保存网格时出错: {e}")
        return False


def load_mesh(file_path):
    """加载网格文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Open3D网格对象
    """
    print(f"加载网格文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        if mesh.is_empty():
            raise ValueError("加载的网格为空")
        
        print(f"网格加载成功，包含 {len(np.asarray(mesh.vertices))} 个顶点")
        return mesh
        
    except Exception as e:
        print(f"加载网格时出错: {e}")
        raise


def convert_o3d_to_pyvista(o3d_mesh):
    """将Open3D网格转换为PyVista网格
    
    Args:
        o3d_mesh: Open3D网格对象
        
    Returns:
        PyVista网格对象
    """
    try:
        # 提取顶点和三角形
        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)
        
        # 如果有法线，也提取出来
        if o3d_mesh.has_vertex_normals():
            normals = np.asarray(o3d_mesh.vertex_normals)
        else:
            normals = None
        
        # 创建PyVista网格
        pv_mesh = pv.PolyData(vertices, triangles)
        
        # 添加法线
        if normals is not None:
            pv_mesh['Normals'] = normals
        
        print(f"成功将Open3D网格转换为PyVista网格: {len(vertices)} 个顶点, {len(triangles)} 个三角形")
        return pv_mesh
        
    except Exception as e:
        print(f"转换Open3D网格到PyVista时出错: {e}")
        raise


def convert_pyvista_to_o3d(pv_mesh):
    """将PyVista网格转换为Open3D网格
    
    Args:
        pv_mesh: PyVista网格对象
        
    Returns:
        Open3D网格对象
    """
    try:
        # 提取顶点和三角形
        vertices = pv_mesh.points
        triangles = pv_mesh.faces.reshape(-1, 4)[:, 1:4]  # PyVista格式: [n_points, idx1, idx2, idx3]
        
        # 创建Open3D网格
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 计算法线
        o3d_mesh.compute_vertex_normals()
        
        print(f"成功将PyVista网格转换为Open3D网格: {len(vertices)} 个顶点, {len(triangles)} 个三角形")
        return o3d_mesh
        
    except Exception as e:
        print(f"转换PyVista网格到Open3D时出错: {e}")
        raise


def calculate_mesh_quality(mesh):
    """计算网格质量指标
    
    Args:
        mesh: Open3D网格对象
        
    Returns:
        包含质量指标的字典
    """
    try:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 计算三角形面积
        def triangle_area(v1, v2, v3):
            return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        
        areas = []
        for tri in triangles:
            v1, v2, v3 = vertices[tri]
            areas.append(triangle_area(v1, v2, v3))
        
        areas = np.array(areas)
        
        # 计算统计信息
        quality = {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "total_area": np.sum(areas),
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "mean_area": np.mean(areas),
            "std_area": np.std(areas)
        }
        
        # 计算边界框
        bbox = {
            "min": np.min(vertices, axis=0),
            "max": np.max(vertices, axis=0),
            "size": np.max(vertices, axis=0) - np.min(vertices, axis=0)
        }
        
        quality["bounding_box"] = bbox
        
        print(f"网格质量计算完成: {quality['triangle_count']} 个三角形, 总面积: {quality['total_area']:.2f}")
        
        return quality
        
    except Exception as e:
        print(f"计算网格质量时出错: {e}")
        return None


def get_mesh_info_string(mesh, name="网格"):
    """获取网格信息的格式化字符串
    
    Args:
        mesh: Open3D网格对象
        name: 网格名称
        
    Returns:
        格式化的信息字符串
    """
    try:
        quality = calculate_mesh_quality(mesh)
        if quality is None:
            return f"{name}: 无法计算质量信息"
        
        info = [
            f"=== {name} 信息 ===",
            f"顶点数量: {quality['vertex_count']:,}",
            f"三角形数量: {quality['triangle_count']:,}",
            f"总面积: {quality['total_area']:.2f}",
            f"三角形面积范围: {quality['min_area']:.6f} - {quality['max_area']:.6f}",
            f"三角形平均面积: {quality['mean_area']:.6f} ± {quality['std_area']:.6f}",
            f"边界框大小: X={quality['bounding_box']['size'][0]:.2f}, "
            f"Y={quality['bounding_box']['size'][1]:.2f}, "
            f"Z={quality['bounding_box']['size'][2]:.2f}"
        ]
        
        return "\n".join(info)
        
    except Exception as e:
        print(f"生成网格信息字符串时出错: {e}")
        return f"{name}: 错误 - {str(e)}"


def create_directory_if_not_exists(directory):
    """如果目录不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        是否成功创建或已存在
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
            return True
        else:
            print(f"目录已存在: {directory}")
            return False
    except Exception as e:
        print(f"创建目录时出错: {e}")
        return False


def format_vector(vector, precision=4):
    """格式化向量为字符串
    
    Args:
        vector: 向量数组
        precision: 小数点精度
        
    Returns:
        格式化的字符串
    """
    format_str = f"[{', '.join([f'%.{precision}f' for _ in vector])}]"
    return format_str % tuple(vector)


def time_function(func):
    """装饰器：测量函数执行时间
    
    Args:
        func: 要测量的函数
        
    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间: {elapsed_time:.4f} 秒")
        return result
    return wrapper


def get_timestamp_filename(prefix="", suffix="", extension=""):
    """生成带时间戳的文件名
    
    Args:
        prefix: 文件名前缀
        suffix: 文件名后缀
        extension: 文件扩展名（不含点）
        
    Returns:
        带时间戳的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix] if prefix else []
    parts.append(timestamp)
    if suffix:
        parts.append(suffix)
    filename = "_".join(parts)
    if extension:
        filename = f"{filename}.{extension}"
    return filename


def compute_center_of_mass(mesh):
    """计算网格的质心
    
    Args:
        mesh: Open3D网格对象
        
    Returns:
        质心坐标
    """
    try:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 计算每个三角形的质心和面积
        centers = []
        areas = []
        
        for tri in triangles:
            v1, v2, v3 = vertices[tri]
            # 三角形质心
            center = (v1 + v2 + v3) / 3.0
            # 三角形面积
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            
            centers.append(center)
            areas.append(area)
        
        centers = np.array(centers)
        areas = np.array(areas)
        
        # 加权平均计算质心
        total_area = np.sum(areas)
        if total_area > 0:
            weighted_center = np.sum(centers * areas[:, np.newaxis], axis=0) / total_area
            return weighted_center
        else:
            # 回退到顶点平均值
            return np.mean(vertices, axis=0)
            
    except Exception as e:
        print(f"计算质心时出错: {e}")
        # 回退到顶点平均值
        return np.mean(np.asarray(mesh.vertices), axis=0)


def normalize_mesh(mesh, target_size=1.0):
    """归一化网格大小
    
    Args:
        mesh: Open3D网格对象
        target_size: 目标大小
        
    Returns:
        归一化后的网格
    """
    try:
        # 计算边界框大小
        vertices = np.asarray(mesh.vertices)
        bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        max_dim = np.max(bbox_size)
        
        # 计算缩放因子
        scale_factor = target_size / max_dim
        
        # 缩放网格
        normalized_mesh = mesh.scale(scale_factor, center=mesh.get_center())
        
        print(f"网格归一化完成，缩放因子: {scale_factor:.4f}")
        return normalized_mesh
        
    except Exception as e:
        print(f"归一化网格时出错: {e}")
        return mesh


def rotate_mesh_to_align_normal(mesh, target_normal=[0, 0, 1]):
    """旋转网格以对齐法线
    
    Args:
        mesh: Open3D网格对象
        target_normal: 目标法线向量
        
    Returns:
        旋转后的网格
    """
    try:
        # 计算网格的主法线方向（通过PCA）
        vertices = np.asarray(mesh.vertices)
        centered = vertices - np.mean(vertices, axis=0)
        covariance = np.dot(centered.T, centered) / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 最大特征值对应的特征向量
        current_normal = eigenvectors[:, 2]
        
        # 确保方向一致
        if np.dot(current_normal, target_normal) < 0:
            current_normal = -current_normal
        
        # 计算旋转矩阵
        def rotation_matrix_from_vectors(vec1, vec2):
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix
        
        rot_matrix = rotation_matrix_from_vectors(current_normal, target_normal)
        
        # 应用旋转
        rotated_mesh = mesh.rotate(rot_matrix, center=mesh.get_center())
        rotated_mesh.compute_vertex_normals()
        
        print("网格旋转以对齐法线完成")
        return rotated_mesh
        
    except Exception as e:
        print(f"旋转网格时出错: {e}")
        return mesh