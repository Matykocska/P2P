import os
import numpy as np
import warnings

data_root = "./data/ShapeNetCore"
split = "train"

split_root = os.path.join(data_root, split)

for folder in os.listdir(split_root):
    print('Checking folder: %s' % folder)
    folder_path = os.path.join(split_root, folder)
    for file in os.listdir(folder_path):
        #print('Checking file: %s' % file)
        if file.endswith(".txt"):
            #print('Checking file: %s' % file)
            file_path = os.path.join(folder_path, file)
            points = [line.rstrip().split() for line in open(file_path)]
            if (len(points) > 8192):
                points = points[:8192]
            if (len(points) < 8192):
                missing = 8192 - len(points)
                points = points + points[:missing]
            save_path = file_path[:-4]
            np.save(save_path, points)
            #print('Saving file: %s' % save_path)