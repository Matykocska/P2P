import os
import numpy as np
import warnings

data_root = "./data/ShapeNetCore"
split = "train"

split_root = os.path.join(data_root, split)
folder_count = 0

for folder in os.listdir(split_root):
    folder_count += 1
    all_points = np.empty([0,8192,3])
    folder_path = os.path.join(split_root, folder)
    file_len = len(os.listdir(folder_path))
    file_count = 0
    all_count = 0
    print('Checking folder: %s %d/55 files: %d' % (folder, folder_count, file_len))
    for file in os.listdir(folder_path):
        file_count += 1
        if file_count % 100 == 0:
            print('  Progress %d/%d' % (file_count, file_len))
        #print('Checking file: %s' % file)
        if file.endswith(".npy") and file.startswith("model"):
            #print('Checking file: %s' % file)
            file_path = os.path.join(folder_path, file)
            points = np.load(file_path)
            points = points.astype(float)
            all_points = np.append(all_points, [points], axis=0)
            # if (all_points.shape[0] == 500):
            #     all_count += 1
            #     save_path = os.path.join(folder_path, '{}_all{}'.format(folder, all_count))
            #     np.save(save_path, all_points)
            #     print('Saving file: %s' % save_path)
            #     all_points = np.empty([0,8192,3])
    save_path = os.path.join(folder_path, '{}_all'.format(folder))
    np.save(save_path, all_points)
    print('Saving file: %s' % save_path)