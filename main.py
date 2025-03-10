import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from src import train_model  
import tkinter.font as tkFont

# Define a custom font (adjust the family name if your system has it installed)
# custom_font = tkFont.Font(family="Helvetica", size=10)  # Or use "Noto Sans CJK JP" if installed

def select_file():
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def update_mode(*args):
    mode = mode_var.get()
    if mode == "Evaluate":
        extra_frame.grid_remove()  # Hide extra parameters
    else:
        extra_frame.grid()         # Show extra parameters

def run_action():
    mode = mode_var.get()
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
        messagebox.showerror("Input Error", "Please check that numeric values are entered correctly.")
        return

    # Update status label to show progress (using thread-safe call)
    root.after(0, lambda: status_label.config(text="Status: Running..."))

    def run_in_thread():
        try:
            if mode == "Train":
                train_model.train_lstm(file_path, n_lags, target_col,
                                       epochs, batch_size, validation_split,
                                       model_save_path)
            elif mode == "Evaluate":
                train_model.evaluate_lstm(file_path, n_lags, target_col,
                                          model_save_path, interval)
            elif mode == "Tune":
                train_model.tune_lstm(file_path, n_lags, target_col,
                                      epochs, batch_size, validation_split,
                                      model_save_path, max_trials=max_trials)
            # On success, update status label via thread-safe call
            root.after(0, lambda: status_label.config(text="Status: Completed"))
        except Exception as e:
            # On error, update status label with error message
            root.after(0, lambda e=e: status_label.config(text=f"Status: Error: {str(e)}"))
        finally:
            # Optionally, clear the status after a few seconds
            root.after(5000, lambda: status_label.config(text="Status: Idle"))

    # Start the operation in a separate thread so the GUI stays responsive
    threading.Thread(target=run_in_thread).start()

# Set up the main window
root = tk.Tk()
root.title("Oil Temperature Prediction")

# Optionally, set a default font for the entire GUI by configuring tk option database
# root.option_add("*Font", custom_font)

# Mode selection
mode_var = tk.StringVar(value="Train")
mode_var.trace_add("write", update_mode)  # Call update_mode whenever mode changes

frame_mode = ttk.LabelFrame(root, text="Select Mode")
frame_mode.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
for i, mode in enumerate(["Train", "Evaluate", "Tune"]):
    ttk.Radiobutton(frame_mode, text=mode, variable=mode_var, value=mode).grid(row=0, column=i, padx=5, pady=5)

# Common parameters frame (always visible)
common_frame = ttk.LabelFrame(root, text="Common Parameters")
common_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

def add_label_entry(parent, label_text, row, default=""):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
    entry = ttk.Entry(parent)
    entry.insert(0, default)
    entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
    return entry

file_entry = add_label_entry(common_frame, "CSV File Path:", 0, "data/ett.csv")
ttk.Button(common_frame, text="Browse", command=select_file).grid(row=0, column=2, padx=5, pady=5)
model_save_entry = add_label_entry(common_frame, "Model Save Path:", 1, "models/ot_model_7d_ft_50.keras")

# Extra parameters frame (visible for Train/Tune modes only)
extra_frame = ttk.LabelFrame(root, text="Extra Parameters")
extra_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

n_lags_entry = add_label_entry(extra_frame, "n_lags:", 0, "5")
target_col_entry = add_label_entry(extra_frame, "Target Column:", 1, "OT")
epochs_entry = add_label_entry(extra_frame, "Epochs:", 2, "20")
batch_size_entry = add_label_entry(extra_frame, "Batch Size:", 3, "32")
validation_split_entry = add_label_entry(extra_frame, "Validation Split:", 4, "0.1")
interval_entry = add_label_entry(extra_frame, "Interval (hours):", 5, "24")
max_trials_entry = add_label_entry(extra_frame, "Max Trials (Tune):", 6, "50")

# Run button
run_button = ttk.Button(root, text="Run", command=run_action)
run_button.grid(row=3, column=0, padx=10, pady=10)

# Status label near the run button
status_label = ttk.Label(root, text="Status: Idle")
status_label.grid(row=4, column=0, padx=10, pady=5)

# Initially update the mode (so that if Evaluate is the default, extra parameters are hidden)
update_mode()

root.mainloop()
