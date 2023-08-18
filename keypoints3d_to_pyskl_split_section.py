import pickle
from pathlib import Path
import os
import numpy as np

# filepath argument
anno_dir = Path('../annotations/keypoints3d')
ignore_file = Path('../annotations/ignore_list.txt')
label_file = Path('../annotations/label.txt')
output_file = Path('../aist++3d_2s.pkl')
split_file_train = Path('../annotations/splits/pose_train.txt')
split_file_val = Path('../annotations/splits/pose_val.txt')
split_file_test = Path('../annotations/splits/pose_test.txt')

# other argument
BM_num_clip = 4


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
        #print("bad file")
        continue

    # set number of clip
    num_clip = None
    if 'sBM' in pkl_filepath.stem:
        num_clip = BM_num_clip
    elif 'sFM' in pkl_filepath.stem:
        num_clip = BM_num_clip*4
    assert num_clip is not None

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

        num_frame_per_clip = keypoints.shape[0] // num_clip
        for i in range(num_clip):
            st_frame = num_frame_per_clip * i
            anno_dict = {}
            anno_dict['keypoint'] = np.expand_dims(keypoints[st_frame:st_frame + num_frame_per_clip], axis=0)
            anno_dict['frame_dir'] = pkl_filepath.stem + f'_{i+1:02d}'
            anno_dict['total_frames'] = anno_dict['keypoint'].shape[1]
            anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])
            
            anno_list.append(anno_dict)
final_dict['annotations'] = anno_list

## Write split
split_dict = {}
with open(split_file_train, 'r') as train_f:
    data = train_f.read()
    org_train_list = data.split('\n')
    new_train_list = []
    for id in org_train_list:
        num_clip = None
        if 'sBM' in id:
            num_clip = BM_num_clip
        elif 'sFM' in id:
            num_clip = BM_num_clip*4
        assert num_clip is not None


        for i in range(num_clip):
            new_train_list.append(id + f'_{i+1:02d}')

    split_dict['train'] = new_train_list


with open(split_file_val, 'r') as val_f:
    data = val_f.read()
    org_val_list = data.split('\n')
    new_val_list = []
    for id in org_val_list:
        num_clip = None
        if 'sBM' in id:
            num_clip = BM_num_clip
        elif 'sFM' in id:
            num_clip = BM_num_clip*4
        assert num_clip is not None

        for i in range(num_clip):
            new_val_list.append(id + f'_{i+1:02d}')

    split_dict['val'] = new_val_list

with open(split_file_test, 'r') as test_f:
    data = test_f.read()
    org_test_list = data.split('\n')
    new_test_list = []
    for id in org_test_list:
        num_clip = None
        if 'sBM' in id:
            num_clip = BM_num_clip
        elif 'sFM' in id:
            num_clip = BM_num_clip*4
        assert num_clip is not None

        for i in range(num_clip):
            new_test_list.append(id + f'_{i+1:02d}')

    split_dict['test'] = new_test_list

final_dict['split'] = split_dict


# Dump into pickle file
with open(output_file, 'wb') as out_f:
    pickle.dump(final_dict, out_f)
