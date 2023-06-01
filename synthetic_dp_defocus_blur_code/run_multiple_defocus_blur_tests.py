import os
import subprocess
import argparse

# コマンドライン引数を解析
parser = argparse.ArgumentParser(description='Run synthetic_dp_defocus_blur_refactor.py with varying fs and fd parameters.')
parser.add_argument('--fs', type=float, nargs='+', help='Values for fs parameter.')
parser.add_argument('--fd', type=int, nargs='+', help='Values for fd parameter.')
args = parser.parse_args()

# 引数からfsとfdの値のリストを取得
fs_values = args.fs if args.fs is not None else [1.4, 2.0, 4.0]
fd_values = args.fd if args.fd is not None else [300, 700, 1000, 1300]

# ベースとなるコマンドを定義（fsとfdの値をプレースホルダーに）
base_command = "python synthetic_dp_defocus_blur_refactor.py -o /data1/teppei_kurita/data/DPSynthesis/output/dp_fuji_fs{fs}_fd{fd}_a1.0000/ -d /data1/teppei_kurita/data/DPSynthesis/input/Fixed3000/ -fd {fd} -fs {fs} -ca 1.0000"

# fsとfdの全ての組み合わせについてコマンドを実行
for fs in fs_values:
    for fd in fd_values:
        # プレースホルダーにfsとfdの値を挿入
        command = base_command.format(fs=fs, fd=fd)
        print("Executing:", command)
        
        # コマンドを実行（subprocess.runを使用）
        process = subprocess.run(command, shell=True, check=True)
