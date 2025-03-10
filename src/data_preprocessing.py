import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_preprocess_data(file_path, n_lags, target_col, interval=1):
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df.set_index('date', inplace=True)

    # Handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Missing values found: {missing_values}. Applying interpolation...")
        df = df.interpolate()
    else:
        print("No missing values found. Skipping interpolation.")

    # Create time-based features BEFORE splitting (safe and can be done upfront)
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday

    def month_to_season(month): # Season feature function
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'autumn'
    df['season'] = df['month'].apply(month_to_season)
    df = pd.get_dummies(df, columns=['season'])
    df[['season_autumn', 'season_spring', 'season_summer', 'season_winter']] = \
        df[['season_autumn', 'season_spring', 'season_summer', 'season_winter']].astype('float32')

    # Split data into training and testing BEFORE scaling and lag feature creation
    X = df # Features will initially include all columns
    y = df[target_col] # Target is still defined as before
    train_size = int(0.8 * len(X))
    X_train_df, X_test_df = X[:train_size], X[train_size:]
    y_train_df, y_test_df = y[:train_size], y[train_size:]

    # --- Scaling AFTER splitting ---
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df) # Fit scaler ONLY on training features
    X_test_scaled = scaler.transform(X_test_df)       # Transform test features using fitted scaler

    # Convert scaled numpy arrays back to DataFrames for feature addition (optional, but cleaner for column access)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_df.columns, index=X_train_df.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_df.columns, index=X_test_df.index)

    # --- Lag features AFTER scaling and splitting ---
    cols_to_lag = ['OT', 'HULL', 'MULL', 'LUFL', 'HUFL', 'MUFL', 'LULL'] # Use original column names as scaling preserves them

    # Lag features for TRAINING set
    for col in cols_to_lag:
        for lag in range(1, n_lags + 1):
            X_train_scaled_df[f'{col}_lag_{lag}'] = X_train_scaled_df[col].shift(lag)
    X_train_scaled_df.dropna(inplace=True)

    # Lag features for TESTING set
    for col in cols_to_lag:
        for lag in range(1, n_lags + 1):
            X_test_scaled_df[f'{col}_lag_{lag}'] = X_test_scaled_df[col].shift(lag)
    X_test_scaled_df.dropna(inplace=True)


    # --- Sequence creation AFTER lag features and splitting ---
    def create_sequences(data_X, data_y, target_col, n_lags, k=interval): # Modified to take X and y
        X, y = [], []
        # Adjust range to account for potential index misalignment after lagging and dropping NaNs.
        # Iterate based on the *intersection* of indices between features and target after NaN removal.
        common_indices = data_X.index.intersection(data_y.index)
        for i in range(n_lags, len(common_indices) - k + 1):
            current_index = common_indices[i]
            X_window_indices = common_indices[i-n_lags:i] # Indices for the window
            X.append(data_X.loc[X_window_indices].values) # Use .loc to select based on index
            y.append(data_y.loc[common_indices[i + k - 1]]) # Target also selected by index.
        return np.array(X), np.array(y)


    # Create sequences for training data using SCALED features and original y_train_df target
    X_train, y_train = create_sequences(X_train_scaled_df, y_train_df, target_col, n_lags, interval)
    # Create sequences for testing data using SCALED features and original y_test_df target
    X_test, y_test = create_sequences(X_test_scaled_df, y_test_df, target_col, n_lags, interval)


    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, scaler, None, df # scaler is fitted only on X_train features, df_scaled is removed for clarity