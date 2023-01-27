import os
import numpy as np
import warnings

data_root = "./data/ShapeNetCore"
split = "train"

split_root = os.path.join(data_root, split)
folder_count = 0

for folder in os.listdir(split_root):
    folder_count += 1
    #all_points = np.empty([0,4096,3])
    folder_path = os.path.join(split_root, folder)
    file_len = len(os.listdir(folder_path))
    #file_count = 0
    #all_count = 0
    print('Checking folder: %s %d/55 files: %d' % (folder, folder_count, file_len))

    file_path = os.path.join(folder_path, '{}_all.npy'.format(folder))
    points = np.load(file_path)
    np.save(os.path.join(folder_path, '{}_all1'.format(folder)), points)

    pt_idxs = np.arange(0, 8192)
    np.random.shuffle(pt_idxs)
    pt_idxs = pt_idxs[:4096]

    save_path = os.path.join(folder_path, '{}_all'.format(folder))
    np.save(save_path, points[:,pt_idxs,:])
    print('Saving file: %s' % save_path)

    # if os.path.isfile(os.path.join(folder_path, '{}_all.npy'.format(folder))):
    #     continue
    # for file in os.listdir(folder_path):
    #     file_count += 1
    #     if file_count % 100 == 0:
    #         print('  Progress %d/%d' % (file_count, file_len))
    #     #print('Checking file: %s' % file)
    #     if file.endswith(".npy") and file.startswith(folder):
    #         #print('Checking file: %s' % file)
    #         file_path = os.path.join(folder_path, file)
    #         points = np.load(file_path)
    #         pt_idxs = np.arange(0, 8192)
    #         np.random.shuffle(pt_idxs)
    #         pt_idxs = pt_idxs[:4096]
    #         all_points = np.append(all_points, points[:,pt_idxs,:], axis=0)
    # save_path = os.path.join(folder_path, '{}_all'.format(folder))
    # np.save(save_path, all_points)
    # print('Saving file: %s' % save_path)