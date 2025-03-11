import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_preprocess_data(file_path, n_lags, target_col, interval=1):
    # データセットを読み込む
    df = pd.read_csv(file_path)

    # 'date'列を日時形式に変換する
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df.set_index('date', inplace=True)

    # 欠損値を処理する
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Missing values found: {missing_values}. Applying interpolation...")
        df = df.interpolate()
    else:
        print("No missing values found. Skipping interpolation.")

    # 分割前に時間ベースの特徴を作成する
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    def month_to_season(month): # 季節の特徴関数
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'autumn'
    df['season'] = df['month'].apply(month_to_season)
    df = pd.get_dummies(df, columns=['season'])
    df[['season_autumn', 'season_spring', 'season_summer', 'season_winter']] = \
        df[['season_autumn', 'season_spring', 'season_summer', 'season_winter']].astype('float32')

    # データをトレーニングとテストに分割する
    X = df # 特徴量は最初はすべての列を含む
    y = df[target_col] # 目標は以前と同じように定義される
    train_size = int(0.8 * len(X))
    X_train_df, X_test_df = X[:train_size], X[train_size:]
    y_train_df, y_test_df = y[:train_size], y[train_size:]

    # 分割後のスケーリング
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df) # トレーニング特徴量のみにスケーラーを適用する
    X_test_scaled = scaler.transform(X_test_df)       # フィットされたスケーラーを使用してテスト特徴量を変換する

    # 特徴量追加のためにスケーリングされたnumpy配列をDataFrameに戻す
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_df.columns, index=X_train_df.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_df.columns, index=X_test_df.index)

    # スケーリングと分割後のラグ特徴
    cols_to_lag = ['OT', 'HULL', 'MULL', 'LUFL', 'HUFL', 'MUFL', 'LULL'] # スケーリングが元の列名を保持するため、元の列名を使用する

    # トレーニングセットのラグ特徴
    for col in cols_to_lag:
        for lag in range(1, n_lags + 1):
            X_train_scaled_df[f'{col}_lag_{lag}'] = X_train_scaled_df[col].shift(lag)
    X_train_scaled_df.dropna(inplace=True)

    # テストセットのラグ特徴
    for col in cols_to_lag:
        for lag in range(1, n_lags + 1):
            X_test_scaled_df[f'{col}_lag_{lag}'] = X_test_scaled_df[col].shift(lag)
    X_test_scaled_df.dropna(inplace=True)


    # ラグ特徴と分割後のシーケンス作成
    def create_sequences(data_X, data_y, target_col, n_lags, k=interval): # Xとyを取るように修正
        X, y = [], []
        # ラグとNaNの削除後のインデックスのずれを考慮して範囲を調整する
        # NaN削除後の特徴量と目標のインデックスの*交差*に基づいて反復する
        common_indices = data_X.index.intersection(data_y.index)
        for i in range(n_lags, len(common_indices) - k + 1):
            current_index = common_indices[i]
            X_window_indices = common_indices[i-n_lags:i] # ウィンドウのインデックス
            X.append(data_X.loc[X_window_indices].values) # インデックスに基づいて選択するために.locを使用する
            y.append(data_y.loc[common_indices[i + k - 1]]) # 目標もインデックスで選択される
        return np.array(X), np.array(y)


    # スケーリングされた特徴量と元のy_train_df目標を使用してトレーニングデータのシーケンスを作成する
    X_train, y_train = create_sequences(X_train_scaled_df, y_train_df, target_col, n_lags, interval)
    # スケーリングされた特徴量と元のy_test_df目標を使用してテストデータのシーケンスを作成する
    X_test, y_test = create_sequences(X_test_scaled_df, y_test_df, target_col, n_lags, interval)


    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, scaler, None, df # スケーラーはX_train特徴量のみにフィットされ、df_scaledは明確さのために削除される