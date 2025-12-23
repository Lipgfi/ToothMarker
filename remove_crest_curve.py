#!/usr/bin/env python3
"""
删除model_viewer.py中与拟合牙槽嵴曲线相关的方法
"""

# 读取文件内容
with open("d:/Edentulous Plane/ui/model_viewer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 定义要删除的方法名称和对应的起始行号
method_info = [
    ("_add_crest_curve_point", 1994),
    ("_show_crest_curve_points", 2019),
    ("_fit_crest_curve", 2072),
    ("_project_crest_curve_to_plane", 2189)
]

# 按行号从大到小排序，避免删除前面的行影响后面的行号
method_info.sort(key=lambda x: x[1], reverse=True)

# 找到每个方法的结束位置并删除
for method_name, start_line in method_info:
    # 转换为0索引
    start_idx = start_line - 1
    end_idx = start_idx
    
    # 找到方法结束位置（下一个def或文件结束）
    for i in range(start_idx, len(lines)):
        if lines[i].strip().startswith("def ") and i > start_idx:
            end_idx = i
            break
    else:
        end_idx = len(lines)
    
    # 删除方法
    del lines[start_idx:end_idx]
    print(f"已删除方法 {method_name}（行 {start_line}-{end_idx+1}）")

# 写入修改后的内容
with open("d:/Edentulous Plane/ui/model_viewer.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("成功删除拟合牙槽嵴曲线相关方法")
