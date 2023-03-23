"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import cv2
import imageio

def normalize_0_1(mat_to_normalize):
    mat_to_normalize=((mat_to_normalize-np.min(mat_to_normalize))/np.max(mat_to_normalize-np.min(mat_to_normalize)))
    return mat_to_normalize

def normalize_scale(mat_to_normalize,max_,min_):
    mat_to_normalize=(max_-min_)/(np.max(mat_to_normalize)-np.min(mat_to_normalize))*(mat_to_normalize-np.min(mat_to_normalize))+min_
    return mat_to_normalize

def create_circle(c_kernel,c_center,c_radius):
    """create a disk of the given radius and center"""
    c_kernel=cv2.circle(c_kernel, c_center, c_radius, (1,1,1), -1)
    return c_kernel

def makeButterworth(bw_size,  cut_off_freq, order_, btype, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    The cut-off frequency $D_0$ is full-width-half-maximum, which
    can be thought of as an effective radius.
    bw_sizeは正方形の一辺の長さ
    カットオフ周波数D_0は、全幅半最大値であり、効果的な半径と考えることができる。
    
    """
    # NumPy 配列を作成
    x = np.arange(0, bw_size, 1, float)
    y = x[:,np.newaxis]
    
    '''assume M=N (square kernel) The cut-off frequency $D_0$ is in the range of 0<D_0<N/2'''
    '''M=N（正方形のカーネル）を仮定している。カットオフ周波数$D_0$は0<D_0<N/2の範囲内'''
    # センターが指定されていない場合、デフォルトで bw_size の半分をセンターとする
    if center is None:
        x0 = y0 = bw_size // 2
    else:
        # センターが指定されている場合、それを使用
        x0 = center[0]
        y0 = center[1]

    # 式の項を計算    
    equ_term=(((x-x0)**2 + (y-y0)**2) / cut_off_freq**2)**order_

    # フィルタータイプが 'low' の場合、ローパスフィルターを適用
    if btype == 'low':
        return 1/(1+equ_term)
    # フィルタータイプが 'high' の場合、ハイパスフィルターを適用
    elif btype == 'high':
        return equ_term/(1+equ_term)


def bw_kernel_generator(k_size_, order_, cut_off_factor_, beta_, smooth_strength_):
    # 円形サイズ行列と中心オフセットを初期化
    circ_size=np.zeros([k_size_,k_size_]) # k_size x k_sizeの配列を生成
    center_offset=(k_size_//2,k_size_//2)

    # 円を作成
    circle=create_circle(circ_size,center_offset,k_size_//2)
        
    # ガウスカーネルサイズを計算
    k_size_gauss=round(k_size_/smooth_strength_)+1
    if k_size_gauss % 2 == 0:
        k_size_gauss+=1

    # ガウスカーネルの標準偏差を計算        
    sigma_gauss = 0.3*((k_size_gauss-1)*0.5 - 1) + 0.8
    padding_gauss=k_size_gauss//2

    # 減衰マスクを作成    
    decay_mask=np.arange(0,k_size_+(2*padding_gauss),1,float)
    decay_mask=decay_mask.reshape((1,len(decay_mask)))
    ones_mask=np.ones([k_size_+(2*padding_gauss),1])
    decay_mask=ones_mask @ decay_mask    

    # バターワースカーネルを作成
    cut_off_=(k_size_-1)/cut_off_factor_
    k_butter=makeButterworth(k_size_,cut_off_,order_,'high') # k_size x k_sizeのフィルターのカーネル

    # 円形マスクとバターワースカーネルを組み合わせ
    k_c=circle*normalize_scale(k_butter,1,beta_)

    # パディングを追加
    k_c_pad=cv2.copyMakeBorder(k_c, padding_gauss, padding_gauss, padding_gauss, padding_gauss, 0)

    # ガウスブラーを適用    
    blur_k_c = cv2.GaussianBlur(k_c_pad,(k_size_gauss,k_size_gauss),sigma_gauss)

    # 減衰マスクを適用
    blur_k_l=blur_k_c*normalize_0_1(decay_mask)

    # 右側のカーネルを作成
    blur_k_r=np.flip(blur_k_l)

    # 正規化して返す
    return blur_k_c/np.sum(blur_k_c), blur_k_l/np.sum(blur_k_l), blur_k_r/np.sum(blur_k_r)