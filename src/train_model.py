import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from . import data_preprocessing
from matplotlib import font_manager as fm
import keras_tuner as kt

def train_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split, model_save_path):
    # データの読み込みと前処理
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # モデルの構築
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # トレーニング
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1)

    # テストセットでの評価
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss (MSE):", test_loss)

    # モデルの保存
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def evaluate_lstm(file_path, n_lags, target_col, model_save_path, interval):
    # 保存されたモデルの読み込み
    model = load_model(model_save_path)
    
    # データの読み込みと前処理
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)
    
    # テストセットでモデルを評価
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss (MSE):", test_loss)
    
    # 予測を行う
    y_pred = model.predict(X_test)
    
    # 日本語フォントを設定
    # フォントプロパティを追加
    jp_font = fm.FontProperties(fname='fonts/NotoSansCJKjp-Regular.ttf')

    # rcParamsを更新
    plt.rcParams['font.family'] = jp_font.get_name()

    # 実際の値と予測値のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='実際 (Actual)')
    plt.plot(y_pred, label='予測 (Predicted)')
    plt.xlabel('時間 (Time)')
    plt.ylabel('油温 (Oil Temperature)')
    plt.title(f'実際の油温と予測油温\nモデル: {model_save_path}\n損失 (平均二乗誤差 - MSE): {test_loss:.8f}')
    plt.legend()
    plt.show()

def tune_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split, model_save_path, max_trials=5):
    # データの読み込みと前処理
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)
    
    # ハイパーパラメータを受け取るモデル構築関数を定義
    def build_model(hp):
        model = Sequential()
        # LSTM層のユニット数を調整
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
        # ドロップアウト率を調整
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        # Adamオプティマイザの学習率を調整
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # RandomSearchチューナーを初期化
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,  # より徹底的な検索のためにこの数を調整できます
        directory='hyperparam_tuning',
        project_name='ot_tuning'
    )

    # ハイパーパラメータ検索を実行
    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    
    # 最適なモデルとハイパーパラメータを取得
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:", best_hp.values)
    
    # 最適なモデルを保存
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")
