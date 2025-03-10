# Oil Temperature Prediction using LSTMs

## Project Structure

*   **`data/`**:  Contains the CSV data file used for training and evaluation. The default file is `ett.csv`.
*   **`models/`**:  Stores pre-trained models. You can save trained models here and load them for evaluation.
*   **`src/`**:  Contains the source code for model training (`train_model.py`) and data preprocessing (`data_preprocessing.py`).
*   **`main.py`**:  The main script that launches the Tkinter GUI application.
*   **`README.md`**: This file, providing instructions and information about the project.
*   **`requirements.txt`**: Lists the Python packages required to run the application.
*   **`fonts/`**: Contains font for displaying Japanese in matplotlib

## How to Run the Application

1.  **Navigate to the project directory** in your terminal or command prompt.
2.  **Activate the virtual environment** if you created one.
3.  **Run the `main.py` script:**
    ```bash
    python main.py
    ```

    This will launch the Oil Temperature Prediction GUI application.

## Using the GUI

The GUI is organized into several sections:

### 1. Select Mode

*   **Train:**  Trains a new LSTM model using the provided data and parameters.
*   **Evaluate:** Evaluates a pre-trained LSTM model (specified by "Model Save Path") on the provided data.
*   **Tune:**  Performs hyperparameter tuning for an LSTM model using Keras Tuner to find the best model configuration based on validation loss.

Select the desired mode using the radio buttons in the "Select Mode" frame. The "Extra Parameters" frame will be shown or hidden depending on whether "Train" or "Tune" mode is selected (it is hidden for "Evaluate" mode).

### 2. Common Parameters

This section is always visible and requires the following inputs:

*   **CSV File Path:**
    *   Enter the path to your CSV data file. You can click the "Browse" button to open a file dialog and select the file. The default is `data/ett.csv`.
*   **Model Save Path:**
    *   Specify the path where the trained model should be saved (for "Train" and "Tune" modes) or the path to the pre-trained model to be evaluated (for "Evaluate" mode). The default path is `models/ot_model_7d_ft_50.keras`.

### 3. Extra Parameters (for Train and Tune Modes)

This section is visible only when "Train" or "Tune" mode is selected. It allows you to configure training and tuning parameters:

*   **n_lags:**
    *   The number of lag time steps to use as input features for the LSTM model. This determines how many past data points the model considers when making a prediction. Default is `5`.
*   **Target Column:**
    *   The name of the column in your CSV file that contains the oil temperature data you want to predict. Default is `OT`.
*   **Epochs:**
    *   The number of training epochs to run. An epoch is one complete pass through the entire training dataset. Default is `20`.
*   **Batch Size:**
    *   The number of samples per gradient update during training. Default is `32`.
*   **Validation Split:**
    *   The fraction of the training data to be used as validation data during training. This is used to monitor the model's performance on unseen data during training. Default is `0.1`.
*   **Interval (hours):**
    *   The prediction interval in hours. For example, if set to `24`, the model will predict oil temperature 24 hours into the future. Default is `24`.
*   **Max Trials (Tune):**
    *   (Only relevant for "Tune" mode) The maximum number of hyperparameter combinations Keras Tuner will try during the tuning process. Increasing this number will likely improve tuning results but will take longer. Default is `50`.


## Installation

1.  **Clone the repository** (if you have the code in a Git repository):
    ```bash
    git clone [repository-url]
    cd [project-directory]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    *   On Windows, activate the environment: `venv\Scripts\activate`
    *   On macOS/Linux, activate the environment: `source venv/bin/activate`

3.  **Install required Python packages:**
    The project includes a `requirements.txt` file listing all necessary Python libraries. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

    This will install pandas, numpy, matplotlib, tensorflow, keras-tuner, scikit-learn, and any other dependencies listed in `requirements.txt`.  *(Note: `tkinter` is usually included with standard Python installations, so you might not need to install it separately.)*