import os
import numpy as np
import warnings

data_root = "./data/ShapeNetCore"

models = [line.rstrip().split(",") for line in open(os.path.join(data_root, "val.csv"))]

#print(models)
npmodels = np.array(models)
#print(models[0])
print(npmodels[0:5])
missing = []
index = 0
model_count = len(models)

all_points = np.empty((0,0,3), np.float32)

for model in models:
    if index % 100 == 0:
        print("{}/{}".format(index,model_count))
    model_points = np.empty((0,3), np.float32)
    try:
        for line in open(os.path.join(data_root, "val", model[1], "model_{}.obj".format(model[0]))):
            model_points
            split_line = line.split()
            if "v" == split_line[0]:
                new_point = np.array([split_line[1], split_line[2], split_line[3]]).astype(np.float32)
                #print(new_point)
                model_points = np.append(model_points, [new_point], axis=0)
        #all_points = np.append(all_points, [model_points], axis=0)
        print(model_points.shape)
        #np.save(os.path.join(data_root, "val_data.npy"), model_points)
    except OSError as e:
        print(e.strerror)
        missing.append(index)
        print(missing)
    index += 1
    if index == 100:
        break
    #print(model_points)

np.delete(npmodels, missing, 0)

np.save(os.path.join(data_root, "val_labels.npy"), npmodels)
np.save(os.path.join(data_root, "val_data.npy"), all_points)