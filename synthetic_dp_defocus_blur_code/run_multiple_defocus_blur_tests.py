import os
import subprocess

# fsとfdの値のリストを定義
fs_values = [1.4, 2.0, 4.0]
fd_values = [300, 700, 1000, 1300]

# ベースとなるコマンドを定義（fsとfdの値をプレースホルダーに）
base_command = "python synthetic_dp_defocus_blur_refactor.py -o ./dp_fuji_fs{fs}_fd{fd}_a1.0000_test/ -d Fixed/ -fd {fd} -fs {fs} -ca 1.0000"

# fsとfdの全ての組み合わせについてコマンドを実行
for fs in fs_values:
    for fd in fd_values:
        # プレースホルダーにfsとfdの値を挿入
        command = base_command.format(fs=fs, fd=fd)
        print("Executing:", command)
        
        # コマンドを実行（subprocess.runを使用）
        process = subprocess.run(command, shell=True, check=True)
