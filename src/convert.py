import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path

# --- パス設定 ---
# 新しいzip形式で保存された.kerasファイルを指定
keras_model_path = Path("/app/data/conducting_model_best.keras") 
# 出力先のフォルダを指定
tfjs_model_path = Path("/app/data/tfjs_model") 

# --- 変換処理 ---
print(f"Kerasモデルを読み込んでいます: {keras_model_path}")

# 1. .kerasファイルをロード
# このload_modelは新しい形式にも対応しています
model = tf.keras.models.load_model(keras_model_path)

print("TensorFlow.js形式への変換を開始します...")

# 2. 読み込んだモデルを直接TF.js形式で保存
tfjs.converters.save_keras_model(model, tfjs_model_path)

print(f"✅ 変換が完了しました！: {tfjs_model_path}")