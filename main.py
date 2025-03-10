import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from src import train_model  
import tkinter.font as tkFont

# Initial language dictionary (English and Japanese)
lang_dict = {
    "select_mode": "Select Mode / モード選択",
    "train": "Train / 訓練",
    "evaluate": "Evaluate / 評価",
    "tune": "Tune / 調整",
    "common_params": "Common Parameters / 共通パラメータ",
    "csv_file_path": "CSV File Path: / CSVファイルパス:",
    "browse": "Browse / ブラウズ",
    "model_save_path": "Model Save Path: / モデル保存パス:",
    "extra_params": "Extra Parameters / 追加パラメータ",
    "n_lags": "n_lags: / 遅れ数:",
    "target_col": "Target Column: / 目標列:",
    "epochs": "Epochs: / エポック数:",
    "batch_size": "Batch Size: / バッチサイズ:",
    "validation_split": "Validation Split: / 検証分割:",
    "interval": "Interval (hours): / 間隔（時間）:",
    "max_trials": "Max Trials (Tune): / 最大試行数（調整）:",
    "run": "Run / 実行",
    "status_idle": "Status: Idle / 状態: 待機中",
    "status_running": "Status: Running... / 状態: 実行中...",
    "status_completed": "Status: Completed / 状態: 完了",
    "status_error": "Status: Error / 状態: エラー",
    "input_error": "Input Error / 入力エラー",
    "check_numeric": "Please check that numeric values are entered correctly. / 数値が正しく入力されているか確認してください。",
    "select_csv": "Select CSV File / CSVファイルを選択"
}

# Mode map for conversion between languages
mode_map = {
    "Train": "Train / 訓練",
    "Evaluate": "Evaluate / 評価",
    "Tune": "Tune / 調整"
}

# Store internal mode (language-independent)
current_internal_mode = "train"  # Default to train mode

def select_file():
    file_path = filedialog.askopenfilename(title=lang_dict["select_csv"], filetypes=[("CSV Files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def update_mode(*args):
    global current_internal_mode
    
    # Get selected mode text
    selected_mode = mode_var.get()
    
    # Map to internal mode ("train", "evaluate", "tune")
    if selected_mode == lang_dict["train"]:
        current_internal_mode = "train"
    elif selected_mode == lang_dict["evaluate"]:
        current_internal_mode = "evaluate"
    elif selected_mode == lang_dict["tune"]:
        current_internal_mode = "tune"
    
    # Update UI based on internal mode
    if current_internal_mode == "evaluate":
        extra_frame.grid_remove()  # Hide extra parameters
    else:
        extra_frame.grid()         # Show extra parameters
    
    # Force geometry recalculation
    root.update_idletasks()
    root.geometry("")  # Reset to natural size

def run_action():
    file_path = file_entry.get()
    model_save_path = model_save_entry.get()
    
    try:
        # Extra parameters (only used for Train/Tune modes)
        n_lags = int(n_lags_entry.get())
        target_col = target_col_entry.get()
        epochs = int(epochs_entry.get())
        batch_size = int(batch_size_entry.get())
        validation_split = float(validation_split_entry.get())
        interval = int(interval_entry.get())
        max_trials = int(max_trials_entry.get())
    except ValueError:
        messagebox.showerror(lang_dict["input_error"], lang_dict["check_numeric"])
        return

    # Update status label to show progress (using thread-safe call)
    root.after(0, lambda: status_label.config(text=lang_dict["status_running"]))

    def run_in_thread():
        try:
            if current_internal_mode == "train":
                train_model.train_lstm(file_path, n_lags, target_col,
                                       epochs, batch_size, validation_split,
                                       model_save_path)
            elif current_internal_mode == "evaluate":
                train_model.evaluate_lstm(file_path, n_lags, target_col,
                                          model_save_path, interval)
            elif current_internal_mode == "tune":
                train_model.tune_lstm(file_path, n_lags, target_col,
                                      epochs, batch_size, validation_split,
                                      model_save_path, max_trials=max_trials)
            # On success, update status label via thread-safe call
            root.after(0, lambda: status_label.config(text=lang_dict["status_completed"]))
        except Exception as e:
            # On error, update status label with error message
            root.after(0, lambda e=e: status_label.config(text=f"{lang_dict['status_error']}: {str(e)}"))
        finally:
            # Optionally, clear the status after a few seconds
            root.after(5000, lambda: status_label.config(text=lang_dict["status_idle"]))

    # Start the operation in a separate thread so the GUI stays responsive
    threading.Thread(target=run_in_thread).start()

def update_labels():
    frame_mode.config(text=lang_dict["select_mode"])
    common_frame.config(text=lang_dict["common_params"])
    file_label.config(text=lang_dict["csv_file_path"])
    browse_button.config(text=lang_dict["browse"])
    model_save_label.config(text=lang_dict["model_save_path"])
    extra_frame.config(text=lang_dict["extra_params"])
    n_lags_label.config(text=lang_dict["n_lags"])
    target_col_label.config(text=lang_dict["target_col"])
    epochs_label.config(text=lang_dict["epochs"])
    batch_size_label.config(text=lang_dict["batch_size"])
    validation_split_label.config(text=lang_dict["validation_split"])
    interval_label.config(text=lang_dict["interval"])
    max_trials_label.config(text=lang_dict["max_trials"])
    run_button.config(text=lang_dict["run"])
    status_label.config(text=lang_dict["status_idle"])

# Set up the main window
root = tk.Tk()
root.title("Oil Temperature Prediction")

# Configure column weights to ensure proper expansion
root.columnconfigure(0, weight=1)

# Mode selection
mode_var = tk.StringVar(value="Train")  # Default value in English
mode_var.trace_add("write", update_mode)  # Call update_mode whenever mode changes

frame_mode = ttk.LabelFrame(root, text="Select Mode / モード選択")
frame_mode.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
frame_mode.columnconfigure(0, weight=1)
frame_mode.columnconfigure(1, weight=1)
frame_mode.columnconfigure(2, weight=1)

# Create mode buttons
mode_buttons = []
for i, mode in enumerate(["Train / 訓練", "Evaluate / 評価", "Tune / 調整"]):  # Default English values
    btn = ttk.Radiobutton(frame_mode, text=mode, variable=mode_var, value=mode)
    btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
    mode_buttons.append(btn)

# Common parameters frame (always visible)
common_frame = ttk.LabelFrame(root, text="Common Parameters / 共通パラメータ")
common_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
common_frame.columnconfigure(1, weight=1)  # Make entry columns expandable

def add_label_entry(parent, label_text, row, default=""):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
    entry = ttk.Entry(parent)
    entry.insert(0, default)
    entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
    return label, entry

file_label, file_entry = add_label_entry(common_frame, "CSV File Path: / CSVファイルパス:", 0, "data/ett.csv")
browse_button = ttk.Button(common_frame, text="Browse / ブラウズ", command=select_file)
browse_button.grid(row=0, column=2, padx=5, pady=5)
model_save_label, model_save_entry = add_label_entry(common_frame, "Model Save Path: / モデル保存パス:", 1, "models/ot_model_7d_ft_50.keras")

# Extra parameters frame (visible for Train/Tune modes only)
extra_frame = ttk.LabelFrame(root, text="Extra Parameters / 追加パラメータ")
extra_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
extra_frame.columnconfigure(1, weight=1)  # Make entry columns expandable

n_lags_label, n_lags_entry = add_label_entry(extra_frame, "n_lags: / 遅れ数:", 0, "5")
target_col_label, target_col_entry = add_label_entry(extra_frame, "Target Column: / 目標列:", 1, "OT")
epochs_label, epochs_entry = add_label_entry(extra_frame, "Epochs: / エポック数:", 2, "20")
batch_size_label, batch_size_entry = add_label_entry(extra_frame, "Batch Size: / バッチサイズ:", 3, "32")
validation_split_label, validation_split_entry = add_label_entry(extra_frame, "Validation Split: / 検証分割:", 4, "0.1")
interval_label, interval_entry = add_label_entry(extra_frame, "Interval (hours): / 間隔（時間）:", 5, "24")
max_trials_label, max_trials_entry = add_label_entry(extra_frame, "Max Trials (Tune): / 最大試行数（調整）:", 6, "50")

# Run button
run_button = ttk.Button(root, text="Run / 実行", command=run_action)
run_button.grid(row=4, column=0, padx=10, pady=10)

# Status label near the run button
status_label = ttk.Label(root, text="Status: Idle / 状態: 待機中")
status_label.grid(row=5, column=0, padx=10, pady=5)

# Initialize default mode
update_mode()

# Set initial language (English by default)
update_labels()

# Allow the window to resize itself to fit contents
root.update_idletasks()
root.geometry("")

root.mainloop()
