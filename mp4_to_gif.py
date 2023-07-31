from moviepy.editor import *
    

input_file = './visualize_result/gBR_sBM_c01_d04_mBR0_ch01_smpl.mp4'
output_file = './visualize_result/gBR_sBM_c01_d04_mBR0_ch01_smpl.gif'

clip = (VideoFileClip(input_file).subclip(0, 3).resize(0.5))
clip.write_gif(output_file, fps=20)