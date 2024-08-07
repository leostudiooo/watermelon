import os
import shutil

prefix = r'/Users/lilingfeng/Repositories/watermelon'

# 原始数据集路径
original_dataset_path = prefix + r'/datasets'
# 预处理后的数据集路径
cleaned_dataset_path = prefix + r'/cleaned'

# 创建预处理后的数据集路径
os.makedirs(cleaned_dataset_path, exist_ok=True)

# original_dataset_path/{experiment#}_{sweetness}/chu/{#}/{wav,jpg} -> cleaned_dataset_path/{sweetness}/重新编号/{*.wav,*.jpg}

subdirs = [d for d in os.listdir(original_dataset_path) if os.path.isdir(
    os.path.join(original_dataset_path, d))]

# 为避免同一糖度的数据集被覆盖，采用全局编号计数器。
global_counter = 0

for subdir in subdirs:
    sweetness = subdir.split('_')[-1].split('\\')[0]

    # go to subdir/chu/[#]
    folders = [f for f in os.listdir(os.path.join(original_dataset_path, subdir, 'chu')) if os.path.isdir(
        os.path.join(original_dataset_path, subdir, 'chu', f))]

    for folder in folders:
        # move, rename wav and jpg files
        currPath = os.path.join(original_dataset_path, subdir, 'chu', folder)
        # create destination folder
        destPath = os.path.join(cleaned_dataset_path, sweetness, str(global_counter))
        os.makedirs(destPath, exist_ok=True)
        shutil.move(os.path.join(currPath, [f for f in os.listdir(currPath) if f.endswith('.wav')][0]), os.path.join(destPath, str(global_counter) + '.wav'))
        shutil.move(os.path.join(currPath, [f for f in os.listdir(currPath) if f.endswith('.jpg')][0]), os.path.join(destPath, str(global_counter) + '.jpg'))
        
        global_counter += 1