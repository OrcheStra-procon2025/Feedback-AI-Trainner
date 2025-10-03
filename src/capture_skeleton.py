import cv2
import mediapipe as mp
import numpy as np
import time

# --- 定数 ---
# AIモデルの入力に合わせて設定
KEY_JOINTS_INDICES = [11, 7, 10, 6]  # 右手, 左手, 右肘, 左肘 (NTU基準)
CAPTURE_DURATION_SEC = 5  # 何秒間キャプチャするか

# --- 初期設定 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0) # 0番のカメラを開く

if not cap.isOpened():
    print("エラー: カメラを開けません。")
    exit()

pose_data_recorder = []
start_time = time.time()
print(f"{CAPTURE_DURATION_SEC}秒間の骨格データキャプチャを開始します...")

# --- メインループ ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("空のフレームを無視します。")
        continue

    # 左右反転し、色をBGRからRGBに変換
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # MediaPipeで骨格推定を実行
    results = pose.process(image)

    # ランドマークが検出された場合
    if results.pose_landmarks:
        # 重要な関節の座標(x, y, z)を抽出
        frame_features = []
        for joint_index in KEY_JOINTS_INDICES:
            landmark = results.pose_landmarks.landmark[joint_index]
            frame_features.extend([landmark.x, landmark.y, landmark.z])
        
        # このフレームのデータを記録
        pose_data_recorder.append(frame_features)

    # 指定時間が経過したらループを抜ける
    if time.time() - start_time > CAPTURE_DURATION_SEC:
        break

# --- 後処理 ---
print("キャプチャを終了し、データを保存します...")
cap.release()

# データをNumPy配列に変換
captured_data = np.array(pose_data_recorder)

# ファイルに保存
output_path = '/app/output/captured_data.npy'
np.save(output_path, captured_data)

print(f"🎉 成功: データを '{output_path}' に保存しました。")
print(f"データ形状: {captured_data.shape}")