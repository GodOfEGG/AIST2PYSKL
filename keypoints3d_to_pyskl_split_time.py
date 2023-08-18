import pickle
from pathlib import Path
import os
import numpy as np

# Path Argument
anno_dir = Path('../annotations/keypoints3d')
ignore_file = Path('../annotations/ignore_list.txt')
label_file = Path('../annotations/label.txt')
output_file = Path('../aist++3d_4s.pkl')
split_file_train = Path('../annotations/splits/pose_train.txt')
split_file_val = Path('../annotations/splits/pose_val.txt')
split_file_test = Path('../annotations/splits/pose_test.txt')

# Other Argument
clip_len = 240      # frame per clip



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
num_clip_dict={}
for pkl_filepath in anno_dir.glob('*.pkl'):
    if pkl_filepath.stem in ignore_file_list:
        continue

    with open(pkl_filepath, 'rb') as pkl_f:
        data = pickle.loads(pkl_f.read())
        

        keypoints = data['keypoints3d']
        new_keypoints = []
        for x in keypoints:
            if False in np.isfinite(x):
                continue
            new_keypoints.append(x)
        keypoints = np.array(new_keypoints)
        assert False not in np.isfinite(keypoints)

        total_len = keypoints.shape[0]
        if clip_len > total_len:
            continue

        num_clip=1
        for i in range(0, total_len, clip_len):
            start_i = i if i+clip_len <= total_len else total_len-clip_len
            end_i = start_i+clip_len
            anno_dict = {}
            anno_dict['keypoint'] = np.expand_dims(keypoints[start_i:end_i], axis=0)
            anno_dict['frame_dir'] = pkl_filepath.stem + f'_{num_clip:02d}'
            anno_dict['total_frames'] = clip_len
            anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])
        
            anno_list.append(anno_dict)
            num_clip+=1
        num_clip_dict[pkl_filepath.stem] = num_clip-1

final_dict['annotations'] = anno_list

## Write split
split_dict = {}
with open(split_file_train, 'r') as train_f:
    data = train_f.read()
    org_train_list = data.split('\n')
    new_train_list = []
    for id in org_train_list:
        if id not in num_clip_dict.keys():
            continue
        num_clip = num_clip_dict[id]

        for i in range(num_clip):
            new_train_list.append(id + f'_{i+1:02d}')

    split_dict['train'] = new_train_list

with open(split_file_val, 'r') as val_f:
    data = val_f.read()
    org_val_list = data.split('\n')
    new_val_list = []
    for id in org_val_list:
        if id not in num_clip_dict.keys():
            continue
        num_clip = num_clip_dict[id]

        for i in range(num_clip):
            new_val_list.append(id + f'_{i+1:02d}')

    split_dict['val'] = new_val_list

with open(split_file_test, 'r') as test_f:
    data = test_f.read()
    org_test_list = data.split('\n')
    new_test_list = []
    for id in org_test_list:
        if id not in num_clip_dict.keys():
            continue
        num_clip = num_clip_dict[id]

        for i in range(num_clip):
            new_test_list.append(id + f'_{i+1:02d}')

    split_dict['test'] = new_test_list

final_dict['split'] = split_dict


# Dump into pickle file
with open(output_file, 'wb') as out_f:
    pickle.dump(final_dict, out_f)
