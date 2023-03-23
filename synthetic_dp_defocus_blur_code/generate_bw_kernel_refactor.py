import numpy as np
import cv2

# 与えられた行列を0から1の範囲に正規化する関数
def normalize_0_1(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    return (mat - min_val) / (max_val - min_val)

# 与えられた行列を指定された範囲に正規化する関数
def normalize_scale(mat, max_val, min_val):
    range_val = max_val - min_val
    return (range_val / (np.max(mat) - np.min(mat))) * (mat - np.min(mat)) + min_val

# 中心と半径を指定して、円を作成する関数
def create_circle(kernel, center, radius):
    return cv2.circle(kernel, center, radius, (1, 1, 1), -1)

# バターワースフィルタを作成する関数
def make_butterworth(size, cutoff_freq, order, btype, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center

    equ_term = (((x - x0) ** 2 + (y - y0) ** 2) / cutoff_freq ** 2) ** order

    if btype == 'low':
        return 1 / (1 + equ_term)
    elif btype == 'high':
        return equ_term / (1 + equ_term)

# ガウシアンカーネルのサイズを計算する関数
def calculate_gaussian_kernel_size(k_size, smooth_strength):
    k_size_gauss = round(k_size / smooth_strength) + 1
    return k_size_gauss + 1 if k_size_gauss % 2 == 0 else k_size_gauss

# ガウシアンカーネルの標準偏差を計算する関数
def calculate_gaussian_std_deviation(k_size_gauss):
    return 0.3 * ((k_size_gauss - 1) * 0.5 - 1) + 0.8

# 減衰マスクを作成する関数
def create_decay_mask(k_size, padding_gauss):
    decay_mask = np.arange(0, k_size + (2 * padding_gauss), 1, float)
    decay_mask = decay_mask.reshape((1, len(decay_mask)))
    ones_mask = np.ones([k_size + (2 * padding_gauss), 1])
    return ones_mask @ decay_mask

# BWカーネルを生成する関数
def bw_kernel_generator(k_size, order, cut_off_factor, beta, smooth_strength):
    circ_size = np.zeros([k_size, k_size])
    center_offset = (k_size // 2, k_size // 2)

    circle = create_circle(circ_size, center_offset, k_size // 2)

    k_size_gauss = calculate_gaussian_kernel_size(k_size, smooth_strength)
    sigma_gauss = calculate_gaussian_std_deviation(k_size_gauss)
    padding_gauss = k_size_gauss // 2

    cut_off = (k_size - 1) / cut_off_factor
    k_butter = make_butterworth(k_size, cut_off, order, 'high')

    # 円とバターワースフィルタの正規化を行う
    k_c = circle * normalize_scale(k_butter, 1, beta)

    # パディングを追加する
    k_c_pad = cv2.copyMakeBorder(k_c, padding_gauss, padding_gauss, padding_gauss, padding_gauss, 0)

    # ガウシアンぼかしを適用する
    blur_k_c = cv2.GaussianBlur(k_c_pad, (k_size_gauss, k_size_gauss), sigma_gauss)

    # 減衰マスクを作成する
    decay_mask = create_decay_mask(k_size, padding_gauss)
    blur_k_l = blur_k_c * normalize_0_1(decay_mask)

    # 右側の減衰マスクを作成する
    blur_k_r = np.flip(blur_k_l)

    # 生成されたカーネルを返す
    return (
        blur_k_c / np.sum(blur_k_c),
        blur_k_l / np.sum(blur_k_l),
        blur_k_r / np.sum(blur_k_r)
    )
