import cv2
import os
import glob
from PIL import Image
    

input_file = '../visualize_result/gBR_sBM_c01_d04_mBR0_ch01_3d_optim.mp4'
output_file = '../visualize_result/gBR_sBM_c01_d04_mBR0_ch01_3d_optim.gif'
gif_len = 180

# read video
cap = cv2.VideoCapture(input_file)
tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok=True)

i=0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(tmp_dir, f'{i:03d}.jpg'),frame)
    i+=1



#write gif
imgs = glob.glob(f'./tmp/*.jpg')
imgs.sort()
pil_frames = [Image.open(img).convert('P') for img in imgs]
frame0 = pil_frames[0]
frame0.save(output_file, format='GIF', append_images=pil_frames[1:gif_len+1], save_all=True, duration=50, loop=0)
