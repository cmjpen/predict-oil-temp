import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import data_preprocessing
from matplotlib import font_manager as fm
import keras_tuner as kt

def train_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split, model_save_path):
    # --- Data Loading and Preprocessing ---  データの読み込みと前処理
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # --- Model Building ---  モデルの構築
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # --- Training ---  トレーニング
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1)

    # --- Evaluation on Test Set ---  テストセットでの評価
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss (MSE):", test_loss)

    # --- Save Model ---  モデルの保存
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def evaluate_lstm(file_path, n_lags, target_col, model_save_path, interval):
    # Load the saved model  保存されたモデルの読み込み
    model = load_model(model_save_path)
    
    # Load and preprocess the data  データの読み込みと前処理
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)
    
    # Evaluate the model on the test set  テストセットでモデルを評価
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss (MSE):", test_loss)
    
    # Make predictions  予測を行う
    y_pred = model.predict(X_test)
    
    # Set jp font
    # Add the font properties
    jp_font = fm.FontProperties(fname='fonts/NotoSansCJKjp-Regular.ttf')

    # Update the rcParams
    plt.rcParams['font.family'] = jp_font.get_name()

    # Plot actual vs predicted values  実際の値と予測値のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='実際 (Actual)')
    plt.plot(y_pred, label='予測 (Predicted)')
    plt.xlabel('時間 (Time)')
    plt.ylabel('油温 (Oil Temperature)')
    plt.title(f'実際の油温と予測油温\nモデル: {model_save_path}\n損失 (平均二乗誤差 - MSE): {test_loss:.8f}')
    plt.legend()
    plt.show()

def tune_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split, model_save_path, max_trials=5):
    # --- Data Loading and Preprocessing ---
    X_train, X_test, y_train, y_test, scaler, df_scaled, df = data_preprocessing.load_and_preprocess_data(file_path, n_lags, target_col)
    
    # Define a model-building function that accepts hyperparameters
    def build_model(hp):
        model = Sequential()
        # Tune the number of units in the LSTM layer
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
        # Tune dropout rate
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        # Tune learning rate for Adam optimizer
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Initialize the RandomSearch tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,  # You can adjust this number for more thorough search
        directory='hyperparam_tuning',
        project_name='ot_tuning'
    )

    # Perform hyperparameter search
    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    
    # Retrieve the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:", best_hp.values)
    
    # Save the best model
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")

if __name__ == "__main__":
    # --- Configuration ---  設定
    file_path = "C:\\Users\\wbscr\\Downloads\\業務体験課題\\assignment-main\\AI-Engineer\\multivariate-time-series-prediction\\ett.csv"
    n_lags = 5
    interval = 1        # predict {interval} hours ahead  --- {interval}時間先を予測
    target_col = 'OT'
    epochs = 20
    batch_size = 32
    validation_split = 0.1
    model_save_path = "ot_model_1h_ft_0.keras"
    
    train = True
    tune = False
    eval = True
    
    # train  トレーニング
    if train:
        train_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split, model_save_path)
    if tune:
        tune_lstm(file_path, n_lags, target_col, epochs, batch_size, validation_split,
                  model_save_path, max_trials=5)
    # evaluate  評価
    if eval:
        evaluate_lstm(file_path, n_lags, target_col, model_save_path, interval)