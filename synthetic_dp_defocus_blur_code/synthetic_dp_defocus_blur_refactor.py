import cv2
import numpy as np
import argparse
import generate_bw_kernel_refactor as bwk #module to generate DP blur kernels
import os
import errno
from wand.image import Image
from pathlib import Path

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Dual-pixel based defocus blur synthesis')
    parser.add_argument('--data_dir', '-d', default='./Chart/', type=str, help='Dataset directory')
    parser.add_argument('--radial_dis', action='store_true', default=False, help='to apply radial distortion or not')
    parser.add_argument('--output_dir', '-o', default='./dp_dataset_sim/', type=str, help='Output directory')
    return parser

def apply_radial_distortion(temp_img, radial_dis_set):
    with Image.from_array(temp_img) as img:
        img.virtual_pixel = 'transparent'
        img.distort('barrel', tuple(radial_dis_set))
        temp_img = np.array(img)[:, :, 0:3]
    return temp_img

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def save_images(images, directory, basename, ext):
    for name, image in images.items():
        cv2.imwrite(str(directory / f"{basename}_{name}{ext}"), image)

def create_sequence_directory_and_save_images(img_name, temp_set, output_dir, max_scene_depth, threshold_dis, sub_img_l, sub_img_r, sub_img_c, sub_depth_l, img_rgb, depth_color_map):
    output_dir = Path(output_dir) / temp_set
    ensure_directory_exists(output_dir)
 
    basename, ext = os.path.splitext(img_name)
    images = {
        'l': sub_img_l,
        'r': sub_img_r,
        'c': sub_img_c,
        'ct': img_rgb,
        'ctd': depth_color_map,
        'ld': (sub_depth_l / max_scene_depth * (2**16-1)).clip(0, 2**16-1).astype(np.uint16),
        'ldc': cv2.applyColorMap(((sub_depth_l / threshold_dis) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    }
    
    save_images(images, output_dir, basename, ext)

def get_directories(data_dir):
    all_dir = [_dir for _dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, _dir))]
    all_dir.sort()
    return all_dir


def get_depth_parameters(data_dir):
    if 'SYNTHIA' in data_dir:
        return 1000, 250
    else:
        return 10, 10


def get_butterworth_parameters():
    bw_para_list = []
    for order in [3, 6, 9]:
        for cut_off_factor in [2.5, 2]:
            for beta in [0.1, 0.2]:
                bw_para_list.append([order, cut_off_factor, beta])
    return bw_para_list


def get_camera_settings(set_name):
    setting_num = 'set_' + str(set_name)
    camera_setting = np.load(setting_num + '.npy')
    focal_len, f_stop, focus_dis = camera_setting
    return focal_len, f_stop, focus_dis


def calculate_lens_parameters(focal_len, f_stop, focus_dis, coc_alpha = 2.0):
    lens_sensor_dis = focal_len * focus_dis / (focus_dis - focal_len)
    lens_dia = focal_len / f_stop
    coc_scale = lens_sensor_dis * lens_dia / focus_dis * 0.5 * coc_alpha
    coc_max = lens_dia * lens_sensor_dis / focus_dis
    return lens_sensor_dis, lens_dia, coc_scale, coc_max


def compute_defocus_map_layers(num_depth_layers, threshold_dis, coc_scale, coc_max, focus_dis):
    # ぼけマップのレイヤーを格納するリストを初期化する
    coc_min_max_dis = []
    ind_count = 0

    # 指定された深度レイヤー数の範囲でループする
    for i in range(num_depth_layers):
        # 最小距離と最大距離を計算する
        min_dis = i / num_depth_layers * threshold_dis
        max_dis = (i + 1) / num_depth_layers * threshold_dis
        sub_dis = (min_dis + (max_dis - min_dis) / 2)

        # coc_size（円形のぼけの大きさ）を計算する
        coc_size = coc_scale * (sub_dis * 1000 - focus_dis) / (sub_dis * 1000)
        coc_size_rounded = round(coc_size)
        coc_size_delta = abs(coc_size_rounded - coc_size)

        # coc_sizeがcoc_maxを超えた場合、coc_maxに制限する
        if abs(coc_size_rounded) > coc_max:
            coc_size_rounded = np.sign(coc_size_rounded) * coc_max

        # iが0より大きい場合の処理
        if i > 0:
            # max_disがthreshold_disに等しい場合、0.1を加算する
            if max_dis == threshold_dis:
                max_dis += 0.1

            # 前の要素のcoc_sizeが現在のcoc_sizeと同じ場合、前の要素のmax_disを更新する
            if coc_min_max_dis[ind_count - 1][0] == coc_size_rounded:
                coc_min_max_dis[ind_count - 1][2] = max_dis
                if coc_min_max_dis[ind_count - 1][3] > coc_size_delta:
                    coc_min_max_dis[ind_count - 1][3] = coc_size_delta
                    coc_min_max_dis[ind_count - 1][4] = sub_dis
            else:
                # そうでない場合、coc_min_max_disに新しい要素を追加する
                coc_min_max_dis.append([int(coc_size_rounded), min_dis, max_dis, coc_size_delta, sub_dis])
                ind_count += 1
        else:
            # iが0の場合、coc_min_max_disに最初の要素を追加する
            coc_min_max_dis.append([int(coc_size_rounded), min_dis, max_dis, coc_size_delta, sub_dis])
            ind_count += 1

    return coc_min_max_dis

def convert_coc_size_to_depth(coc_size, focus_dis, lens_sensor_dis, lens_dia):
    return abs((lens_dia * lens_sensor_dis * focus_dis) / (coc_size * focus_dis - lens_dia * lens_sensor_dis))

def compute_defocus_map_layers_v2(coc_min, coc_max, threshold_dis, coc_scale, focus_dis, lens_sensor_dis, lens_dia):
    # ぼけマップのレイヤーを格納するリストを初期化する
    coc_min_max_dis = []
    ind_count = 0

    # 指定された深度レイヤー数の範囲でループする
    for i in range(coc_min, int(coc_max+1)):
        # 最小距離と最大距離を計算する
        min_dis = convert_coc_size_to_depth(i, focus_dis, lens_sensor_dis, lens_dia)
        if i==int(coc_max):
            max_dis = threshold_dis*1000
        else:
            max_dis = convert_coc_size_to_depth(i+1, focus_dis, lens_sensor_dis, lens_dia)
        coc_size_integer = i

        # coc_min_max_disに新しい要素を追加する
        coc_min_max_dis.append([coc_size_integer, min_dis, max_dis])
        ind_count += 1

    return coc_min_max_dis


def load_images_from_directory(dir_name, prefix, suffixes):
    return [os.path.join(dir_name, prefix, f) for f in os.listdir(os.path.join(dir_name, prefix))
            if f.endswith(suffixes)]


def load_image_and_depth(image_path, depth_path, data_dir, max_scene_depth, threshold_dis):
    # 画像を読み込む
    img_rgb = cv2.imread(image_path)
    # 深度画像を読み込む
    depth = (cv2.imread(depth_path, -1)).astype(np.float64)

    # SYNTHIAデータセットの場合、深度の値を変換する
    if 'SYNTHIA' in data_dir:
        depth = max_scene_depth * (depth[:, :, 2] + depth[:, :, 1] * 256 + depth[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    else:
        # それ以外の場合、深度の値を変換する
        depth = max_scene_depth * ((2 ** 16 - 1) - depth[:, :, 0]) / (2 ** 16)

    # 深度がしきい値を超えた場合、しきい値に制限する
    depth = np.where((depth > threshold_dis), threshold_dis, depth)

    # 深度カラーマップを作成する
    depth_color_map = depth / threshold_dis
    depth_color_map = cv2.applyColorMap((depth_color_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # 3チャンネルの深度画像を作成する
    depth = np.stack((depth, depth, depth), axis=2)

    return img_rgb, depth, depth_color_map


def process_coc_layers(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength):
    # サブ画像リストを初期化する
    sub_imgs_l = []
    sub_imgs_r = []
    sub_imgs_c = []
    # 深度セットを初期化する
    depth_set = []
    # サブ深度リストを初期化する
    sub_depths_l = []
    # カーネルリストを初期化する
    kernels = []

    # coc_min_max_disの各要素に対して処理を行う
    for i, (coc_size, min_dis, max_dis, coc_size_delta, sub_dis) in enumerate(coc_min_max_dis):
        print('coc_size:', coc_size, 'min_dis:', min_dis, 'max_dis:', max_dis, 'coc_size_delta', coc_size_delta, 'sub_dis:', sub_dis)

        # 指定した範囲の深度を持つピクセルにマスクを適用する
        sub_depth = (np.where((depth >= min_dis) & (depth < max_dis), 1, 0)).astype(np.uint8)
        sub_depth_matt = np.where((depth >= min_dis) & (depth < max_dis), 1, matting_ratio)

        # サブ画像を計算する
        sub_img = (img_rgb.astype(np.float16) * sub_depth_matt).astype(np.uint8)
        depth_set.append(sub_depth)

        # coc_sizeが0の場合、サブ画像はすべて同じ
        if coc_size == 0:
            sub_img_l = sub_img_r = sub_img_c = sub_img
        else:
            # coc_sizeが負の場合、カーネルを生成する
            if coc_size < 0:
                kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(2 * abs(coc_size) + 1, order, cut_off_factor, beta, smooth_strength)
            else:
                kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(2 * abs(coc_size) + 1, order, cut_off_factor, beta, smooth_strength)

            # カーネルを適用してサブ画像を生成する
            sub_img_l = cv2.filter2D(sub_img, -1, kernel_l)
            sub_img_r = cv2.filter2D(sub_img, -1, kernel_r)
            sub_img_c = cv2.filter2D(sub_img, -1, kernel_c)

        # サブ画像をリストに追加する
        sub_imgs_l.append(sub_img_l)
        sub_imgs_r.append(sub_img_r)
        sub_imgs_c.append(sub_img_c)

        # サブ深度値を計算する
        sub_depth_value = (depth.astype(np.float16) * sub_depth_matt)
        sub_depth_l = cv2.filter2D(sub_depth_value[:, :, 0], -1, kernel_l)
        sub_depths_l.append(sub_depth_l)
        
    return sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l

def generate_sub_images(img_rgb, coc_size, sub_dis, kernel_l, kernel_r, kernel_c):
    # 入力画像とsub_disを要素ごとに乗算し、データ型をnp.uint8に変換する
    sub_img = (img_rgb.astype(np.float16) * sub_dis).astype(np.uint8)

    # coc_sizeが0の場合、すべてのサブ画像が同じであるため、同じサブ画像を3つ返す
    if coc_size == 0:
        return sub_img, sub_img, sub_img
    else:
        # それ以外の場合、各カーネルを適用したサブ画像を返す
        return cv2.filter2D(sub_img, -1, kernel_l), cv2.filter2D(sub_img, -1, kernel_r), cv2.filter2D(sub_img, -1, kernel_c)

def process_coc_layers_blend(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength):
    # 各層のデータを処理する関数
    def process_coc_layer(layer_data, depth, img_rgb):
        # 各層のデータを展開
        coc_size, min_dis, max_dis, _, sub_dis = layer_data
        # 指定した範囲の深度を持つピクセルにマスクを適用
        sub_depth = (np.where((depth >= min_dis) & (depth < max_dis), 1, 0)).astype(np.uint8)
        # 距離の差を計算
        distance_sub_dis_depth = depth - sub_dis
        # マスクを適用
        sub_depth_matt = np.where((depth >= min_dis) & (depth < max_dis), 1, matting_ratio)
        # サブ深度値を計算
        sub_depth_value = (depth.astype(np.float16) * sub_depth_matt)

        # カーネルを生成
        if coc_size < 0:
            kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(2 * abs(coc_size) + 1, order, cut_off_factor, beta, smooth_strength)
        else:
            kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(2 * abs(coc_size) + 1, order, cut_off_factor, beta, smooth_strength)

        # サブ画像を生成
        sub_img_l, sub_img_r, sub_img_c = generate_sub_images(img_rgb, coc_size, sub_depth_matt, kernel_l, kernel_r, kernel_c)
        # サブ深度値にカーネルを適用
        sub_depth_l = cv2.filter2D(sub_depth_value[:, :, 0], -1, kernel_l)

        return sub_img_l, sub_img_r, sub_img_c, sub_depth, sub_depth_l, distance_sub_dis_depth

    sub_images = []
    distances = []

    # 各層のデータを処理
    for i, layer_data in enumerate(coc_min_max_dis):
        print(layer_data)
        sub_img_l, sub_img_r, sub_img_c, sub_depth, sub_depth_l, distance_sub_dis_depth = process_coc_layer(layer_data, depth, img_rgb)
        sub_images.append((sub_img_l, sub_img_r, sub_img_c, sub_depth, sub_depth_l))
        distances.append(distance_sub_dis_depth)

    # ブレンド処理
    for i, sub_img_data in enumerate(sub_images):
        sub_img_l, sub_img_r, sub_img_c, sub_depth, sub_depth_l = sub_img_data
        distance_sub_dis_depth = distances[i]

        # 前後のサブ画像データを取得
        if i > 0:
            sub_img_l_prev, sub_img_r_prev, sub_img_c_prev, _, _ = sub_images[i - 1]
            distance_sub_dis_depth_prev = distances[i - 1]
        else:
            sub_img_l_prev, sub_img_r_prev, sub_img_c_prev = sub_img_l, sub_img_r, sub_img_c
            distance_sub_dis_depth_prev = distance_sub_dis_depth

        if i < len(sub_images) - 1:
            sub_img_l_next, sub_img_r_next, sub_img_c_next, _, _ = sub_images[i + 1]
            distance_sub_dis_depth_next = distances[i + 1]
        else:
            sub_img_l_next, sub_img_r_next, sub_img_c_next = sub_img_l, sub_img_r, sub_img_c
            distance_sub_dis_depth_next = distance_sub_dis_depth

        # ブレンドアルファを計算
        distance_sub_dis_depth_alt = np.where((distance_sub_dis_depth >= 0), distance_sub_dis_depth_next, distance_sub_dis_depth_prev)
        blend_alpha = np.abs(distance_sub_dis_depth_alt) / (np.abs(distance_sub_dis_depth) + np.abs(distance_sub_dis_depth_alt))

        # 前後のサブ画像を選択
        sub_img_l_alt = np.where((distance_sub_dis_depth >= 0), sub_img_l_next, sub_img_l_prev)
        sub_img_r_alt = np.where((distance_sub_dis_depth >= 0), sub_img_r_next, sub_img_r_prev)
        sub_img_c_alt = np.where((distance_sub_dis_depth >= 0), sub_img_c_next, sub_img_c_prev)

        # サブ画像をブレンド
        sub_img_l = sub_img_l * blend_alpha + sub_img_l_alt * (1.0 - blend_alpha)
        sub_img_r = sub_img_r * blend_alpha + sub_img_r_alt * (1.0 - blend_alpha)
        sub_img_c = sub_img_c * blend_alpha + sub_img_c_alt * (1.0 - blend_alpha)

        sub_images[i] = sub_img_l, sub_img_r, sub_img_c, sub_depth, sub_depth_l

    # 結果をまとめる
    sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l = zip(*sub_images)

    return sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l

def nearest_odd_integer(a):
    """
    与えられた数値aに対して、aの2倍した値bが必ず奇数の整数cになるように計算する。
    cはbと最も近い奇数の整数として選ばれる。

    :param a: float, 入力の数値。整数に限らない。
    :return: int, 計算された奇数の整数c。
    """
    b = a * 2
    if b % 2 == 0:
        c = b + 1
    else:
        c = round(b)
        if c % 2 == 0:
            if b - c < c + 1 - b:
                c -= 1
            else:
                c += 1
    return int(c)


def process_coc_layers_v2(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength, coc_scale, focus_dis):
    # サブ画像リストを初期化する
    sub_imgs_l = []
    sub_imgs_r = []
    sub_imgs_c = []
    # 深度セットを初期化する
    depth_set = []
    # サブ深度リストを初期化する
    sub_depths_l = []
    # カーネルリストを初期化する
    kernels = []

    depth_mm = depth*1000

    coc_size = coc_scale * (depth_mm - focus_dis) / depth_mm

    # coc_min_max_disの各要素に対して処理を行う
    for i, (coc_size_integer, min_dis, max_dis) in enumerate(coc_min_max_dis):
        print('coc_size_integer:', coc_size_integer, 'min_dis:', min_dis, 'max_dis:', max_dis)

        # 指定した範囲の深度を持つピクセルにマスクを適用する
        sub_depth = (np.where((depth_mm >= min_dis) & (depth_mm < max_dis), 1, 0)).astype(np.uint8)
        sub_depth_matt = np.where((depth_mm >= min_dis) & (depth_mm < max_dis), 1, matting_ratio)

        # サブ画像を計算する
        sub_img = (img_rgb.astype(np.float16) * sub_depth_matt).astype(np.uint8)
        depth_set.append(sub_depth)

        if i == 0:
            # coc_sizeが0の場合、サブ画像はすべて同じ
            if coc_size_integer == 0:
                sub_img_l = sub_img_r = sub_img_c = sub_img
            else:
                # coc_sizeの符号によってカーネルl,rを反転させる
                if coc_size_integer < 0:
                    # kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(2 * abs(coc_size_integer) + 1, order, cut_off_factor, beta, smooth_strength)
                    kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(nearest_odd_integer(abs(coc_size_integer)), order, cut_off_factor, beta, smooth_strength)
                else:
                    # kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(2 * abs(coc_size_integer) + 1, order, cut_off_factor, beta, smooth_strength)
                    kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(nearest_odd_integer(abs(coc_size_integer)) + 1, order, cut_off_factor, beta, smooth_strength)

                # カーネルを適用してサブ画像を生成する
                sub_img_l = cv2.filter2D(sub_img, -1, kernel_l)
                sub_img_r = cv2.filter2D(sub_img, -1, kernel_r)
                sub_img_c = cv2.filter2D(sub_img, -1, kernel_c)
        else:
            sub_img_l = sub_img_l_next
            sub_img_r = sub_img_r_next
            sub_img_c = sub_img_c_next

        coc_size_integer_next, min_dis_next, max_dis_next = coc_min_max_dis[min(i+1, len(coc_min_max_dis)-1)]

        if coc_size_integer_next == 0:
            sub_img_l_next = sub_img_r_next = sub_img_c_next = sub_img
        else:
            # coc_sizeの符号によってカーネルl,rを反転させる
            if coc_size_integer_next < 0:
                # kernel_c_next, kernel_r_next, kernel_l_next = bwk.bw_kernel_generator(2 * abs(coc_size_integer_next) + 1, order, cut_off_factor, beta, smooth_strength)
                kernel_c_next, kernel_r_next, kernel_l_next = bwk.bw_kernel_generator(nearest_odd_integer(abs(coc_size_integer_next)), order, cut_off_factor, beta, smooth_strength)
            else:
                # kernel_c_next, kernel_l_next, kernel_r_next = bwk.bw_kernel_generator(2 * abs(coc_size_integer_next) + 1, order, cut_off_factor, beta, smooth_strength)
                kernel_c_next, kernel_l_next, kernel_r_next = bwk.bw_kernel_generator(nearest_odd_integer(abs(coc_size_integer_next)), order, cut_off_factor, beta, smooth_strength)

            # カーネルを適用してサブ画像を生成する
            sub_img_l_next = cv2.filter2D(sub_img, -1, kernel_l_next)
            sub_img_r_next = cv2.filter2D(sub_img, -1, kernel_r_next)
            sub_img_c_next = cv2.filter2D(sub_img, -1, kernel_c_next)

        # ブレンドアルファを計算
        distance_coc_size = coc_size - coc_size_integer
        blend_alpha = 1.0 - distance_coc_size

        # サブ画像をブレンド
        sub_img_l = sub_img_l * blend_alpha + sub_img_l_next * (1.0 - blend_alpha)
        sub_img_r = sub_img_r * blend_alpha + sub_img_r_next * (1.0 - blend_alpha)
        sub_img_c = sub_img_c * blend_alpha + sub_img_c_next * (1.0 - blend_alpha)

        # サブ画像をリストに追加する
        sub_imgs_l.append(sub_img_l)
        sub_imgs_r.append(sub_img_r)
        sub_imgs_c.append(sub_img_c)

        # サブ深度値を計算する
        sub_depth_value = (depth.astype(np.float16) * sub_depth_matt)
        sub_depth_l = cv2.filter2D(sub_depth_value[:, :, 0], -1, kernel_l)
        sub_depths_l.append(sub_depth_l)
        
    return sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l


def combine_sub_images(sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l, num_coc_layers):
    sub_img_l = sub_imgs_l[num_coc_layers - 1] * depth_set[num_coc_layers - 1]
    sub_img_r = sub_imgs_r[num_coc_layers - 1] * depth_set[num_coc_layers - 1]
    sub_img_c = sub_imgs_c[num_coc_layers - 1] * depth_set[num_coc_layers - 1]
    sub_depth_l = sub_depths_l[num_coc_layers - 1] * depth_set[num_coc_layers - 1][:, :, 0]

    for i in range(num_coc_layers - 1):
        index = num_coc_layers - 2 - i
        sub_img_l += sub_imgs_l[index] * depth_set[index]
        sub_img_r += sub_imgs_r[index] * depth_set[index]
        sub_img_c += sub_imgs_c[index] * depth_set[index]
        sub_depth_l += sub_depths_l[index] * depth_set[index][:, :, 0]

    return sub_img_l, sub_img_r, sub_img_c, sub_depth_l


def apply_radial_distortion_to_all(images, radial_dis_set):
    distorted_images = []
    for image in images:
        distorted_images.append(apply_radial_distortion(image, radial_dis_set))
    return distorted_images


def print_parameters(set_name, focal_len, f_stop, focus_dis, lens_sensor_dis, lens_dia, coc_scale):
    print(f'set: {set_name}\nfocal_len: {focal_len}\nf_stop: {f_stop}\nfocus_dis: {focus_dis}\n'
          f'lens_sensor_dis: {lens_sensor_dis}\nlens_dia: {lens_dia}\ncoc_scale: {coc_scale}')

def load_images(data_dir, dir_name):
    image_suffixes = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    if 'SYNTHIA' in data_dir:
        images_rgb = load_images_from_directory(dir_name, 'RGBLeft', image_suffixes)
        images_depth = load_images_from_directory(dir_name, 'DepthLeft', image_suffixes)
    elif 'SIM' in data_dir:
        images_rgb = load_images_from_directory(dir_name, 'polar', ('s0_denoised.png',))
        images_depth = load_images_from_directory(dir_name, 'polar', ('gtD.png',))
    elif 'Chart' in data_dir:
        # images_rgb = load_images_from_directory(dir_name, '', ('s0.png',))
        # images_depth = load_images_from_directory(dir_name, '', ('gtD.png',))
        # dis = 125
        dis = 80
        images_rgb = [dir_name + '/' + 'ID-'+str(dis-1)+'_CZ-0_LZ-30_CA-0_LA--90_CD-'+str(dis)+'_CP--1_LD-60_LP--1_LR-0_RX-0_RY-0_RZ-0_s0.png']
        images_depth = [dir_name + '/' + 'ID-'+str(dis-1)+'_CZ-0_LZ-30_CA-0_LA--90_CD-'+str(dis)+'_CP--1_LD-60_LP--1_LR-0_RX-0_RY-0_RZ-0_gtD.png']
    else:
        images_rgb = load_images_from_directory(dir_name, '', ('s0.png',))
        images_depth = load_images_from_directory(dir_name, '', ('gtD.png',))

    images_rgb.sort()
    images_depth.sort()

    return images_rgb, images_depth



def process_set_name(set_name, num_depth_layers, threshold_dis, coc_alpha, coc_min):
    # ディレクトリカウントを初期化する
    dir_count = 0
    # カメラ設定を取得する
    focal_len, f_stop, focus_dis = get_camera_settings(set_name)
    # レンズパラメータを計算する
    lens_sensor_dis, lens_dia, coc_scale, coc_max = calculate_lens_parameters(focal_len, f_stop, focus_dis, coc_alpha)
    # パラメータを表示する
    print_parameters(set_name, focal_len, f_stop, focus_dis, lens_sensor_dis, lens_dia, coc_scale)
    # ぼけマップのレイヤーを計算する
    # coc_min_max_dis = compute_defocus_map_layers(num_depth_layers, threshold_dis, coc_scale, coc_max, focus_dis)
    coc_min_max_dis = compute_defocus_map_layers_v2(coc_min, coc_max, threshold_dis, coc_scale, focus_dis, lens_sensor_dis, lens_dia)
    return coc_scale, coc_max, focus_dis, coc_min_max_dis

def process_image_pair(img_rgb_path, img_depth_path, data_dir, max_scene_depth, threshold_dis, num_depth_layers, matting_ratio, order, cut_off_factor, beta, smooth_strength, radial_dis, output_dir, coc_min_max_dis, num_coc_layers, coc_scale, focus_dis):
    # 画像のパスを表示する
    print(img_rgb_path)
    # 画像と深度データを読み込む
    img_rgb, depth, depth_color_map = load_image_and_depth(img_rgb_path, img_depth_path, data_dir, max_scene_depth, threshold_dis)
    # 深度データの最小値を表示する
    print('depth_min:', np.min(depth))
    # cocレイヤーを処理する
    # sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l = process_coc_layers(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength)
    # sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l = process_coc_layers_blend(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength)
    sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l = process_coc_layers_v2(img_rgb, depth, coc_min_max_dis, matting_ratio, order, cut_off_factor, beta, smooth_strength, coc_scale, focus_dis)
    # サブ画像を組み合わせる
    sub_img_l, sub_img_r, sub_img_c, sub_depth_l = combine_sub_images(sub_imgs_l, sub_imgs_r, sub_imgs_c, depth_set, sub_depths_l, num_coc_layers)
    # 放射状の歪みがある場合、適用する
    if radial_dis:
        sub_img_l, sub_img_r, sub_img_c, sub_depth_l, img_rgb, depth_color_map = apply_radial_distortion_to_all([sub_img_l, sub_img_r, sub_img_c, sub_depth_l, img_rgb, depth_color_map], radial_dis_set)
    # 画像名を取得する
    img_name = os.path.basename(img_rgb_path)
    # シーケンスディレクトリを作成し、画像を保存する
    create_sequence_directory_and_save_images(img_name, 'train', output_dir, max_scene_depth, threshold_dis, sub_img_l, sub_img_r, sub_img_c, sub_depth_l, img_rgb, depth_color_map)

def main():
    # データディレクトリを設定する
    data_dir = args.data_dir
    # すべてのディレクトリを取得する
    all_dir = get_directories(data_dir)
    # 深度レイヤーの数を設定する
    num_depth_layers = 2000
    # cos_sizeのminを設定する
    coc_min= -10
    # マッティング比率を設定する
    matting_ratio = 1
    # 深度パラメータを取得する
    max_scene_depth, threshold_dis = get_depth_parameters(data_dir)
    # スムージング強度を設定する
    smooth_strength = 7
    # バターワースパラメータを取得する
    bw_para_list = get_butterworth_parameters()
    # 放射状の距離を設定する
    radial_dis = args.radial_dis
    # ポストセットを設定する
    post_set = '_bw_rd' if radial_dis else '_bw'
    # cocのスケーリングパラメータを設定する
    coc_alpha = 2.0
    # セット名を設定する
    set_names = ['canon']

    # セット名ごとに処理を行う
    for set_name in set_names:
        coc_scale, coc_max, focus_dis, coc_min_max_dis = process_set_name(set_name, num_depth_layers, threshold_dis, coc_alpha, coc_min)
        num_coc_layers = len(coc_min_max_dis)

        # シーケンス数とディレクトリ数を初期化する
        seq_count, dir_count = 0, 0
        # すべてのディレクトリを処理する
        for _dir in all_dir:
            seq_count += 1
            dir_name = os.path.join(data_dir, _dir)
            print(dir_count, '   ', dir_name)
            order, cut_off_factor, beta = bw_para_list[0]
            print('order:', order, ', cut_off_factor:', cut_off_factor, ', beta:', beta)
            dir_count += 1
            # 画像を読み込む
            images_rgb_path, images_depth_path = load_images(data_dir, dir_name)

            # 各画像ペアを処理する
            for j in range(len(images_rgb_path)):
                process_image_pair(images_rgb_path[j], images_depth_path[j], data_dir, max_scene_depth, threshold_dis, num_depth_layers, matting_ratio, order, cut_off_factor, beta, smooth_strength, radial_dis, args.output_dir, coc_min_max_dis, num_coc_layers, coc_scale, focus_dis)

if __name__ == '__main__':
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    main()