# Procon/src/create_lstm_dataset.py (改良版)

import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# --- 定数定義 ---
# 右手, 左手, 右肘, 左肘, 右肩, 左肩 (相対座標と速度を計算する対象)
KEY_JOINTS_INDICES = [11, 7, 10, 6, 8, 4] 
# 基準点となる関節 (体幹の中心)
ROOT_JOINT_INDEX = 1
# 1つの動画の最大フレーム数
MAX_TIMESTEPS = 150

# --- パス設定 ---
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
skeleton_folder = project_root / 'data' / 'skeletons'
output_data_path = project_root / 'data' / 'lstm_data_enhanced.npy'
output_labels_path = project_root / 'data' / 'lstm_labels_enhanced.npy'
output_scaler_path = project_root / 'data' / 'lstm_scaler_enhanced.gz'

# (parse_ntu_skeleton関数は変更なし)
def parse_ntu_skeleton(filepath):
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        frame_count = int(lines[0])
        all_frames_data, line_idx = [], 1
        for _ in range(frame_count):
            if line_idx >= len(lines): break
            body_count = int(lines[line_idx]); line_idx += 1
            frame_data = np.zeros((25, 3))
            if body_count > 0:
                line_idx += 1
                joint_count = int(lines[line_idx]); line_idx += 1
                for j in range(joint_count):
                    joint_info = lines[line_idx].split()
                    frame_data[j] = [float(coord) for coord in joint_info[0:3]]
                    line_idx += 1
                if body_count > 1: line_idx += (body_count - 1) * (1 + 1 + joint_count)
            all_frames_data.append(frame_data)
        return np.array(all_frames_data)
    except (IOError, ValueError): return None

# --- メイン処理 ---
def main():
    skeleton_files = [f for f in os.listdir(skeleton_folder) if f.endswith('.skeleton')]
    
    all_sequences = []
    all_labels = []

    for i, filename in enumerate(skeleton_files):
        print(f"処理中 ({i+1}/{len(skeleton_files)}): {filename}")
        filepath = skeleton_folder / filename
        
        skeleton_data = parse_ntu_skeleton(filepath)
        if skeleton_data is None or len(skeleton_data) == 0: continue

        # 1. 重要な関節と基準点のデータを抽出
        key_joints_data = skeleton_data[:, KEY_JOINTS_INDICES, :]
        root_joint_data = skeleton_data[:, [ROOT_JOINT_INDEX], :] # 形状維持のためリストで指定

        # 2. 相対座標の計算 (ブロードキャストを利用)
        relative_coords = key_joints_data - root_joint_data

        # 3. 速度の計算 (フレーム間の差分)
        # np.diffは(N-1)個の差分を計算するので、先頭にゼロを追加して長さを合わせる
        velocity = np.diff(key_joints_data, axis=0)
        velocity = np.concatenate([np.zeros((1, velocity.shape[1], velocity.shape[2])), velocity], axis=0)

        # 4. 特徴量を結合し、2次元配列に変形
        # (フレーム数, 関節数, 3) -> (フレーム数, 関節数 * 3)
        relative_coords_flat = relative_coords.reshape(len(relative_coords), -1)
        velocity_flat = velocity.reshape(len(velocity), -1)
        # (相対座標の特徴量, 速度の特徴量) を横に結合
        sequence = np.concatenate([relative_coords_flat, velocity_flat], axis=1)

        # 5. パディング/トランケーションで長さを揃える
        padded_sequence = np.zeros((MAX_TIMESTEPS, sequence.shape[1]))
        length = min(len(sequence), MAX_TIMESTEPS)
        padded_sequence[:length, :] = sequence[:length, :]
        all_sequences.append(padded_sequence)

        # 6. ラベルの作成
        try:
            label = int(filename.split('A')[1][:3]) - 1
            all_labels.append(label)
        except (IndexError, ValueError):
            del all_sequences[-1]

    X = np.array(all_sequences)
    y = np.array(all_labels)

    # 7. データ全体をスケーリング
    X_reshaped = X.reshape(-1, X.shape[2])
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(X.shape)

    # 8. ファイルに保存
    np.save(output_data_path, X_scaled)
    np.save(output_labels_path, y)
    joblib.dump(scaler, output_scaler_path)

    print("-" * 30)
    print(f"🎉 高度な特徴量を持つデータセットの作成が完了！")
    print(f"特徴量数: {X_scaled.shape[2]}")
    print(f"データ保存先: {output_data_path}")

if __name__ == '__main__':
    main()