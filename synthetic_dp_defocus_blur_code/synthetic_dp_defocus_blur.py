"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

This module is used to apply synthetic defocus blur based on depth map.

Copyright (c) 2020-present, Abdullah Abuolaim

This code applies synthetic defocus blur based on depth map and following the
dual-pixel (DP) image formation. In particular, it synthesizes the DP views
independently based on the spatially varying defocus blur and thin lens model.

v1:
1. optimized the performance of applying the blur kernel based on precomputed
defocus map.
2. imported and applied estimated blur kernels from real cameras.
1.デフォーカスマップの事前計算によるぼかしカーネル適用性能の最適化 最適化した。
2. 実カメラから推定されたぼかしカーネルを取り込み、適用した。
Email: abuolaim@eecs.yorku.ca
"""
import cv2
import numpy as np
import argparse
import generate_bw_kernel as bwk #module to generate DP blur kernels
import os
import errno
from wand.image import Image

def apply_radial_distortion(_temp_img,_radial_dis_set):
    with Image.from_array(_temp_img) as _img:
        _img.virtual_pixel = 'transparent'
        _img.distort('barrel', (_radial_dis_set[0], _radial_dis_set[1], _radial_dis_set[2], _radial_dis_set[3]))
        # convert to opencv/numpy array format
        _temp_img = (np.array(_img))[:,:,0:3]
    return _temp_img
    
def check_dir(path_):
    '''Check directory if exist, if not, create directory using given path'''
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def create_seq_dir_write_img(temp_set, output_dir='./dd_dp_dataset_synth_vid/'):
    '''synthetic video dataset hierarchy. check and create direcorty for each image sequence'''
    # 合成映像データセットの階層化。各映像シーケンスに対応したディレクトリの確認と作成。'
    dir_l_src = output_dir+temp_set+'_l/source/seq_'+str(seq_count).zfill(3)+'/'
    dir_r_src = output_dir+temp_set+'_r/source/seq_'+str(seq_count).zfill(3)+'/'
    dir_c_src = output_dir+temp_set+'_c/source/seq_'+str(seq_count).zfill(3)+'/'
    dir_c_trg = output_dir+temp_set+'_c/target/seq_'+str(seq_count).zfill(3)+'/'
    '''uncomment the following lines in case you need to save depth map'''
    dir_c_trg_d = output_dir+temp_set+'_c/target_d/seq_'+str(seq_count).zfill(3)+'/'
    
    check_dir(dir_l_src)
    check_dir(dir_r_src)
    check_dir(dir_c_src)
    check_dir(dir_c_trg)
    check_dir(dir_c_trg_d)
    
    '''writing output images'''
    cv2.imwrite(dir_l_src+img_name, sub_img_l)
    cv2.imwrite(dir_r_src+img_name, sub_img_r)
    cv2.imwrite(dir_c_src+img_name, sub_img_c)
    cv2.imwrite(dir_c_trg+img_name, img_rgb)
    cv2.imwrite(dir_c_trg_d+img_name, depth_color_map)

    # depth LR
    dir_l_src_d = output_dir+temp_set+'_l/source_d/seq_'+str(seq_count).zfill(3)+'/'
    dir_l_src_d_color = output_dir+temp_set+'_l/source_d_color/seq_'+str(seq_count).zfill(3)+'/'
    check_dir(dir_l_src_d)
    check_dir(dir_l_src_d_color)
    sub_depth_l_color=sub_depth_l/threshold_dis
    sub_depth_l_color=cv2.applyColorMap((sub_depth_l_color*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    sub_depth_l_out = sub_depth_l / max_scene_depth * (2**16-1)
    sub_depth_l_out = np.clip(sub_depth_l_out, 0, 2**16-1)
    cv2.imwrite(dir_l_src_d+img_name, sub_depth_l_out.astype(np.uint16))
    cv2.imwrite(dir_l_src_d_color+img_name, sub_depth_l_color)


parser = argparse.ArgumentParser(description='Dual-pixel based defocus blur synthesis')
'''You can download SYNTHIA-SF dataset from: https://synthia-dataset.net/downloads/'''
parser.add_argument('--data_dir', default='../SYNTHIA-SF/', type=str,  help='SYNTHIA-SF dataset directory')
parser.add_argument('--radial_dis', action='store_true', default=False, help='to apply radial distortion or not')
parser.add_argument('--output_dir', default='./dp_dataset_tynthia/', type=str,  help='Output directory')
args = parser.parse_args()

###############################################
#SYNTHIA-SF Dataset directories and parameters#
###############################################                
data_dir=args.data_dir
all_dir=[ _dir for _dir in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, _dir)) ]
all_dir.sort()

num_depth_layers=2000 # number of discrete depth layers
matting_ratio=1 # weight used for composing image layers in the final stage 最終段階で画像レイヤーを構成するために使用される重み

"""Max distance of SYNTHIA dataset is 5000m, we need a threshold of 250m"""
if 'SYNTHIA' in data_dir:
    max_scene_depth=1000 # maximum scene distance in real world, in m
    threshold_dis=250 #distance threshold for maximum allowed distance
else:
    max_scene_depth=10 # maximum scene distance in real world, in m
    threshold_dis=10 #distance threshold for maximum allowed distance

###############################################
#Butterworth Filter parameters                #
###############################################
smooth_strength=7 # this is realted to kappa in the main paper as kappa = 1/smooth_strength
bw_para_list=[]

for order in [3, 6, 9]:
    for cut_off_factor in [2.5,2]:
        for beta in [0.1, 0.2]:
            bw_para_list.append([order,cut_off_factor,beta])

###############################################
#Radial distortion                            #
###############################################
radial_dis= args.radial_dis

################################################
#Output directory type: estimated, log, or moon#
################################################
if radial_dis:
    post_set='_bw_rd'
else:
    post_set='_bw'

seq_count = 0
for set_num in range(5):
    dir_count = 0 # directory count, each directory represents an image sequence
    ###############################################
    #Camera settings and thin lens model equations#
    ###############################################
    setting_num='set_'+str(set_num)
    coc_max=40 #set the maximum circle of confusion size (i.e., blur kernel size) 錯乱円の最大サイズを設定する
    
    camera_setting=np.load(setting_num+'.npy')
    radial_dis_set=np.load(setting_num+'_rd.npy')
    
    focal_len=camera_setting[0] #camera focal length in mm
    f_stop=camera_setting[1] #camera aperture size in f-stops
    focus_dis= camera_setting[2] #distance between lens and focal plane in mm
    
    #distance between the lens and imaging sensor レンズと撮像素子の距離
    lens_sensor_dis=focal_len*focus_dis/(focus_dis-focal_len)
    lens_dia=focal_len/f_stop #lens diameter in mm

    # # Debug cam parameter
    # print('set:', set_num)
    # print('focal_len:', focal_len)
    # print('f_stop:', f_stop)
    # print('focus_dis:', focus_dis)
    # print('lens_sensor_dis:', lens_sensor_dis)
    # print('lens_dia:', lens_dia)
    # continue
    
    #thin lens model. Scale used to determine the coc size based on thin lens model
    # 薄型レンズモデル。薄型レンズモデルから錯乱円のサイズを決定する際に使用したスケール
    coc_scale=lens_sensor_dis*lens_dia/focus_dis
    
    
    ###############################################
    #Precompute defocus map layers                #
    ###############################################
    coc_min_max_dis=[]
    ind_count=0
    for i in range(num_depth_layers):
        min_dis=i/num_depth_layers*threshold_dis
        max_dis=(i+1)/num_depth_layers*threshold_dis
        sub_dis=(min_dis+(max_dis-min_dis)/2)
        
        #calculate coc size based on thin lens model
        coc_size=round(coc_scale*(sub_dis-focus_dis)/sub_dis)
        if abs(coc_size) > coc_max:
            coc_size=np.sign(coc_size)*coc_max
        if i > 0:
            if max_dis==threshold_dis:
                max_dis+=0.1
            if coc_min_max_dis[ind_count-1][0] == coc_size:
                coc_min_max_dis[ind_count-1][2] = max_dis
            else:
                coc_min_max_dis.append([int(coc_size),min_dis,max_dis])
                ind_count+=1
        else:
            coc_min_max_dis.append([int(coc_size),min_dis,max_dis])
            ind_count+=1
    
    #################################################
    #Framework to apply synthetic defocus blur based#
    #on thin lens model and DP image formation      #
    # 薄型レンズモデルとDP結像に基づく合成焦点ぼかしの適用フレームワーク      
    #################################################
    for _dir in all_dir:
        seq_count += 1
        dir_name=data_dir+_dir + '/'
        print(dir_count,'   ',dir_name)
        order,cut_off_factor,beta = bw_para_list[dir_count%len(bw_para_list)]
        dir_count+=1
        if 'SYNTHIA' in data_dir:
            images_rgb = [dir_name + 'RGBLeft/' + f for f in os.listdir(dir_name + 'RGBLeft/')
                        if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
            images_depth = [dir_name + 'DepthLeft/' + f for f in os.listdir(dir_name + 'DepthLeft/')
                            if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
        else:
            images_rgb = [dir_name + 'polar/' + f for f in os.listdir(dir_name + 'polar/')
                        if f.endswith(('s0_denoised.png'))]
            images_depth = [dir_name + 'polar/' + f for f in os.listdir(dir_name + 'polar/')
                        if f.endswith(('gtD.png'))]
        images_rgb.sort()
        images_depth.sort()
        
        for j in range(len(images_rgb)): #read image by image (i.e., video frames)
            img_rgb=cv2.imread(images_rgb[j])
            depth=(cv2.imread(images_depth[j],-1)).astype(np.float64)
            """
            The following depth manipulation steps are for depth encoding of SYNTHIA dataset.
            There is also depth clipping in order to get more fine quantized depth intervals.
            以下の深度操作手順は、SYNTHIAデータセットの深度エンコーディングのためのものである。
            また、より細かく量子化された深度間隔を得るために、深度クリッピングも行っている。
            """
            if 'SYNTHIA' in data_dir:
                depth=max_scene_depth * (depth[:,:,2] + depth[:,:,1]*256 + depth[:,:,0]*256*256) / (256*256*256 - 1)
            else:
                depth=max_scene_depth * depth[:,:,0] / (2**16-1)
    
            depth=np.where((depth>threshold_dis),threshold_dis,depth)
            
            depth_color_map=depth/threshold_dis
            depth_color_map=cv2.applyColorMap((depth_color_map*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            depth=np.stack((depth,depth,depth), axis=2) # 1 ch to 3 chs, replicate
            
            '''list to keep sub-image and sub-depth'''
            sub_imgs_l=[]
            sub_imgs_r=[]
            sub_imgs_c=[]
            depth_set=[]

            sub_depths_l=[]
            
            img_name=images_rgb[j].split('/')[-1] #extract image name from path
            
            """
            Loop to quantize continuous depth map into equal depth layer intervals.
            連続した深度マップを等しい深度レイヤー間隔に量子化するループ。
            """
            num_coc_layers=len(coc_min_max_dis)
            for i in range(num_coc_layers):
                coc_size=coc_min_max_dis[i][0]
                min_dis=coc_min_max_dis[i][1]
                max_dis=coc_min_max_dis[i][2]
                                
                sub_depth=(np.where((depth >= min_dis) & (depth < max_dis), 1, 0)).astype(np.uint8)
                sub_depth_matt=np.where((depth >= min_dis) & (depth < max_dis), 1, matting_ratio)
                
                '''subimage based on depth layer, weighted by matting_ratio'''
                sub_img=(img_rgb.astype(np.float16)*sub_depth_matt).astype(np.uint8)
                depth_set.append(sub_depth)

                if coc_size == 0:
                        coc_size=1                
                if coc_size < 0:
                    '''generate the blur kernel for DP views based on coc size'''
                    # order: 3, 6, 9
                    # cut_off: 2.5, 2.0
                    # beta: 0.1, 0.2
                    kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(2*abs(coc_size)+1, order, cut_off_factor, beta, smooth_strength)
                else:
                    '''generate the blur kernel for DP views based on coc size'''
                    kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(2*abs(coc_size)+1, order, cut_off_factor, beta, smooth_strength)
        
                '''synthesize DP left view. Convolve subimage with predefined kernel'''
                sub_img_l= cv2.filter2D(sub_img,-1,kernel_l)
                '''synthesize DP right view. Convolve subimage with predefined kernel'''
                sub_img_r= cv2.filter2D(sub_img,-1,kernel_r)
                '''combined final output image'''
                sub_img_c= cv2.filter2D(sub_img,-1,kernel_c)
                
                sub_imgs_l.append(sub_img_l)
                sub_imgs_r.append(sub_img_r)
                sub_imgs_c.append(sub_img_c)

                # Depth LR
                sub_depth_value=(depth.astype(np.float16)*sub_depth_matt)
                sub_depth_l = cv2.filter2D(sub_depth_value[:,:,0],-1,kernel_l)
                sub_depths_l.append(sub_depth_l)

            """
            Composing image layers. This step can be done in the above loop, but it is
            separated here for developing and debugging purposes.
            """
            sub_img_l=sub_imgs_l[num_coc_layers-1]*depth_set[num_coc_layers-1]
            sub_img_r=sub_imgs_r[num_coc_layers-1]*depth_set[num_coc_layers-1]
            sub_img_c=sub_imgs_c[num_coc_layers-1]*depth_set[num_coc_layers-1]
            sub_depth_l=sub_depths_l[num_coc_layers-1]*depth_set[num_coc_layers-1][:,:,0]
            for i in range(num_coc_layers-1):
                sub_img_l+=sub_imgs_l[num_coc_layers-2-i]*depth_set[num_coc_layers-2-i]
                sub_img_r+=sub_imgs_r[num_coc_layers-2-i]*depth_set[num_coc_layers-2-i]
                sub_img_c+=sub_imgs_c[num_coc_layers-2-i]*depth_set[num_coc_layers-2-i]
                sub_depth_l+=sub_depths_l[num_coc_layers-2-i]*depth_set[num_coc_layers-2-i][:,:,0]
            # print(sub_depth_l)

            
            if radial_dis:
                print('aaa')
                sub_img_l= apply_radial_distortion(sub_img_l,radial_dis_set)
                sub_img_r= apply_radial_distortion(sub_img_r,radial_dis_set)
                sub_img_c= apply_radial_distortion(sub_img_c,radial_dis_set)
                sub_depth_l= apply_radial_distortion(sub_depth_l,radial_dis_set)
                img_rgb= apply_radial_distortion(img_rgb,radial_dis_set)
                depth_color_map= apply_radial_distortion(depth_color_map,radial_dis_set)
            # print(sub_depth_l)
            
            if dir_count == 3:
                create_seq_dir_write_img('val', args.output_dir)
                create_seq_dir_write_img('test', args.output_dir)
            else:
                create_seq_dir_write_img('train', args.output_dir)