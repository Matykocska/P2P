import os
import numpy as np
import warnings
import pickle

import torch

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapeNetCore(torch.utils.data.Dataset):
    def __init__(self, config, split):
        self.root = config.data_root
        #self.use_normals = config.use_normals
        self.num_category = config.classes
        self.split = split

        self.all_points = np.empty([0,config.npoints,3])
        self.list_of_labels = []
        # self.labels = [line.rstrip() for line in open(os.path.join(self.root, 'shapenetcore_{}.csv'.format(split)))]

        split_root = os.path.join(self.root, split)
        label_count = 0

        for folder in os.listdir(split_root):
            print('Checking folder: %s' % folder)
            folder_path = os.path.join(split_root, folder)
            for file in os.listdir(folder_path):
                #print('Checking file: %s' % file)
                if file.endswith("all.npy"):
                    print('Checking file: %s' % file)
                    file_path = os.path.join(folder_path, file)
                    points = np.load(file_path)
                    self.all_points = np.append(self.all_points, points, axis=0)
                    for i in range(points.shape[0]):
                        self.list_of_labels.append(label_count)
                    print('Added %d new labels: %s, all: %d' % (points.shape[0], folder, len(self.list_of_labels)))
                    # for label in self.labels:
                    #     if label[0] == file[:-4]:
                    #         self.list_of_labels.append(label[1])
            label_count += 1

        #self.list_of_labels = np.array(self.list_of_labels).astype(np.int64)
        #self.list_of_labels = torch.from_numpy(self.list_of_labels)
        print('The size of %s data is %d == %d' % (split, len(self.list_of_labels), self.all_points.shape[0]))
        #print(self.list_of_labels[0])
        #print(self.all_points[0])

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.all_points[index], self.list_of_labels[index]

        point_set = pc_normalize(point_set)
        point_set = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
        return point_set, label

    def __getitem__(self, index):
        #print("getitem index: %d" % index)
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        #label = label.squeeze(0)
        #print("Label: %s" % label)
        #print("Shape: ")
        #print(current_points.shape)
        return current_points, label