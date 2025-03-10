# オイル温度予測 (LSTMを使用)

このプロジェクトは、電気変圧器から得られる様々なパラメータを基に、LSTM（長短期記憶ニューラルネットワーク）を用いてオイル温度を予測する時系列予測プロジェクトです。予測精度向上のため、ハイパーパラメータチューニングを活用しています。データセットは、[ETDatasetのGitHubリポジトリ](https://github.com/zhouhaoyi/ETDataset) にて公開されているものを使用しています。

## **プロジェクト構成**

- **`data/`**:  学習および評価に使用するCSVデータファイルを保存します。デフォルトのファイルは `ett.csv` です。  
- **`models/`**:  事前学習済みのモデルを保存します。学習後のモデルをここに保存し、評価時に読み込むことができます。  
- **`src/`**:  モデルの学習 (`train_model.py`) およびデータ前処理 (`data_preprocessing.py`) のソースコードが含まれています。  
- **`main.py`**:  Tkinterを使用したGUIアプリケーションを起動するメインスクリプトです。  
- **`README.md`**:  プロジェクトの説明と使用方法を記載したファイルです。  
- **`requirements.txt`**:  必要なPythonパッケージを記載したファイルです。  
- **`fonts/`**:  matplotlibで日本語フォントを表示するためのフォントファイルを含みます。  

## **アプリケーションの実行方法**

1. **ターミナルまたはコマンドプロンプトでプロジェクトディレクトリに移動**  
2. **仮想環境を使用する場合は、仮想環境を有効化**  
3. **`main.py` を実行**  
    ```bash
    python main.py
    ```  
    上記のコマンドを実行すると、オイル温度予測のGUIアプリケーションが起動します。  

## **GUIの使い方**

### **1. モードの選択**

- **Train（学習）**:  指定されたデータとパラメータを使用して、新しいLSTMモデルを学習します。  
- **Evaluate（評価）**:  事前学習済みのLSTMモデル（"Model Save Path" に指定されたもの）を使用して、データを評価します。  
- **Tune（ハイパーパラメータ調整）**:  Keras Tuner を使用して、LSTMモデルの最適な構成を探索し、バリデーション損失を最小化するパラメータを見つけます。  

「Select Mode（モード選択）」フレームでラジオボタンを使用して、目的のモードを選択してください。「Train」または「Tune」を選択すると、「Extra Parameters」セクションが表示され、「Evaluate」を選択すると非表示になります。  

### **2. 共通パラメータ**

このセクションは常に表示され、以下の入力が必要です。  

- **CSVファイルのパス:**  
    - 学習・評価に使用するCSVデータのパスを入力します。「Browse」ボタンをクリックすると、ファイル選択ダイアログが開き、ファイルを選択できます。デフォルトは `data/ett.csv` です。  
- **モデル保存パス:**  
    - 学習済みモデルの保存先（"Train" や "Tune" モード時）または評価用の事前学習済みモデルのパス（"Evaluate" モード時）を指定します。デフォルトは `models/ot_model_7d_ft_50.keras` です。  

### **3. 追加パラメータ（Train・Tuneモード専用）**

「Train」または「Tune」モードを選択した場合のみ、このセクションが表示されます。以下のパラメータを設定できます。  

- **n_lags（遅れデータ数）:**  
    - LSTMモデルの入力として使用する過去のデータポイント数を指定します。デフォルトは `5` です。  
- **ターゲット列:**  
    - 予測対象となるオイル温度データが含まれるCSVファイルの列名を指定します。デフォルトは `OT` です。  
- **エポック数:**  
    - 学習時のエポック（データ全体を何回繰り返して学習するか）を指定します。デフォルトは `20` です。  
- **バッチサイズ:**  
    - 1回の学習で処理するデータのサンプル数を指定します。デフォルトは `32` です。  
- **バリデーション分割:**  
    - 学習データのうち、バリデーション用に使用する割合を指定します。デフォルトは `0.1` です。  
- **予測間隔（時間単位）:**  
    - 予測する時間間隔を指定します。例えば、`24` に設定すると、24時間後のオイル温度を予測します。デフォルトは `24` です。  
- **最大試行回数（Tune専用）:**  
    - ("Tune" モード時のみ有効) Keras Tuner がハイパーパラメータの組み合わせを試す最大回数を指定します。デフォルトは `50` です。  

## **インストール方法**

Python 3.10.0を使います。

1. **リポジトリをクローン（Gitを使用する場合）**  
    ```bash
    git clone https://github.com/cmjpen/predict-oil-temp.git
    cd predict-oil-temp
    ```  

2. **仮想環境を作成（推奨）**  
    ```bash
    python -m venv venv
    ```  
    - Windowsの場合: `venv\Scripts\activate`  
    - macOS/Linuxの場合: `source venv/bin/activate`  

3. **必要なPythonパッケージをインストール**  
    `requirements.txt` に記載されているライブラリをインストールします。  
    ```bash
    pip install -r requirements.txt
    ```  
    これにより、`pandas`、`numpy`、`matplotlib`、`tensorflow`、`keras-tuner`、`scikit-learn` などの依存関係がインストールされます。  
    *(`tkinter` は標準のPython環境に含まれているため、別途インストールは不要な場合が多いです。)*  
