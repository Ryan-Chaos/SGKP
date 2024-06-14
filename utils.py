import os
import shutil

# 指定的文件夹路径
folder_path1 = "D:/work/data/SECOND_val_set/im1"
folder_path2 = "D:/work/data/SECOND_val_set/im2"
destination_folder = "D:/work/data/SECOND_val_set/outputs"

# 要添加的前缀
prefix1 = "1_"
prefix2 = "2_"

# 选择所有以 .SuperPoint+Boost-F 结尾的文件
for filename in os.listdir(folder_path1):
    if filename.endswith(".SuperPoint+Boost-F"):
    # if filename.endswith(".SuperPoint"):
        # 在文件名前添加前缀
        new_filename = prefix1 + filename
        # 构建源文件和目标文件的完整路径
        original_file = os.path.join(folder_path1, filename)
        new_file_path = os.path.join(destination_folder, new_filename)
        # 剪切文件到新的文件夹
        shutil.move(original_file, new_file_path)

for filename in os.listdir(folder_path2):
    if filename.endswith(".SuperPoint+Boost-F"):
    # if filename.endswith(".SuperPoint"):
        # 在文件名前添加前缀
        new_filename = prefix2 + filename
        # 构建源文件和目标文件的完整路径
        original_file = os.path.join(folder_path2, filename)
        new_file_path = os.path.join(destination_folder, new_filename)
        # 剪切文件到新的文件夹
        shutil.move(original_file, new_file_path)

print("文件已经被成功重命名并移动到指定文件夹。")