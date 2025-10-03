# Procon/src/train.py (改良版)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- 💡RTX 40シリーズの性能を最大限に引き出す設定 ---
# 混合精度学習を有効化 (VRAM削減 & 高速化)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- パスとハイパーパラメータ設定 ---
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
data_path = project_root / 'data' / 'lstm_data_enhanced.npy'
labels_path = project_root / 'data' / 'lstm_labels_enhanced.npy'
model_save_path = project_root / 'data' / 'conducting_model_best.keras' # 最良モデルの保存先

# ハイパーパラメータ
EPOCHS = 100 # EarlyStoppingがあるので多めに設定
BATCH_SIZE = 64 # VRAM使用量に応じて調整 (32, 64, 128など)
VALIDATION_SPLIT = 0.2

# --- メイン処理 ---
def main():
    # 1. データ読み込み
    print(f"データセットを読み込んでいます: {data_path}")
    X = np.load(data_path)
    y = np.load(labels_path)
    
    num_samples, timesteps, num_features = X.shape
    num_classes = len(np.unique(y))
    print(f"データ読み込み完了。形状: {X.shape}, クラス数: {num_classes}")

    # 2. データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 💡双方向LSTMモデルの構築
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, num_features)),
        BatchNormalization(),
        Dropout(0.5),
        
        Bidirectional(LSTM(128)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # 出力層の活性化関数はfloat32である必要があるため、dtypeを指定
        Dense(num_classes, activation='softmax', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 4. 💡効率的な学習のためのコールバック設定
    callbacks = [
        # 10エポック性能が改善しなければ早期終了
        EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True),
        # 検証データの損失が最小のモデルを自動保存
        ModelCheckpoint(filepath=model_save_path, save_best_only=True, monitor='val_loss', verbose=1),
        # 5エポック性能が改善しなければ学習率を半分に
        ReduceLROnPlateau(patience=5, monitor='val_loss', factor=0.5, verbose=1)
    ]

    # 5. モデルの学習
    print("\nモデルの学習を開始します...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks
    )

    # 6. モデルの評価 (EarlyStoppingが最良の重みを復元してくれる)
    print("-" * 30)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"テストデータでの正解率 (Accuracy): {accuracy:.4f}")
    print(f"🎉 最良モデルが {model_save_path} に保存されました。")

if __name__ == '__main__':
    main()