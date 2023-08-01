import pickle
from pathlib import Path
import os
import numpy as np


anno_dir = Path('../annotations/keypoints3d')
ignore_file = Path('../annotations/ignore_list.txt')
label_file = Path('../annotations/label.txt')
output_file = Path('../aist++3d_full_choreo.pkl')
split_file_train = Path('../annotations/splits/pose_train.txt')
split_file_val = Path('../annotations/splits/pose_val.txt')
split_file_test = Path('../annotations/splits/pose_test.txt')

## Read label file
label_list = None
with open(label_file, 'r') as label_f:
    data = label_f.read()
    label_list = data.split('\n')
print(label_list)


# Read ignore list to remove bad data
ignore_file_list = []
with open(ignore_file, 'r') as ignore_f:
    data = ignore_f.read()
    ignore_file_list = data.split('\n')


## Write annotation list
final_dict={}
anno_list = []
for pkl_filepath in anno_dir.glob('*.pkl'):
    if pkl_filepath.stem in ignore_file_list:
        print("bad file")
        continue
    if 'sFM' in pkl_filepath.stem:
        continue
    with open(pkl_filepath, 'rb') as pkl_f:
        data = pickle.loads(pkl_f.read())
        anno_dict = {}

        keypoints = data['keypoints3d']
        new_keypoints = []
        for x in keypoints:
            if False in np.isfinite(x):
                continue
            new_keypoints.append(x)
        keypoints = np.array(new_keypoints)
        assert False not in np.isfinite(keypoints)

        anno_dict['keypoint'] = np.expand_dims(keypoints, axis=0)
        anno_dict['frame_dir'] = pkl_filepath.stem
        anno_dict['total_frames'] = keypoints.shape[0]
        anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3]) * 10 + int(pkl_filepath.stem[-2:]) - 1
        

        anno_list.append(anno_dict)
final_dict['annotations'] = anno_list

## Write split
split_dict = {}
with open(split_file_train, 'r') as train_f:
    data = train_f.read()
    split_dict['train'] = data.split('\n')

with open(split_file_val, 'r') as val_f:
    data = val_f.read()
    split_dict['val'] = data.split('\n')

with open(split_file_test, 'r') as test_f:
    data = test_f.read()
    split_dict['test'] = data.split('\n')

final_dict['split'] = split_dict


# Dump into pickle file
with open(output_file, 'wb') as out_f:
    pickle.dump(final_dict, out_f)
