import pickle
import os

pkl_file_path = "annotations\keypoints3d\gBR_sBM_cAll_d04_mBR0_ch01.pkl"

with open(pkl_file_path, 'rb') as f:
    data = pickle.loads(f.read())
    print(data['keypoints3d'].shape)