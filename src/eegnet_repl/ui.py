from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox
from tkinter.ttk import Progressbar

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from mne.viz import plot_topomap
import mne

from eegnet_repl.config import Paths
from eegnet_repl.logger import logger
from eegnet_repl.model import EEGNet


def load_model(path: Path) -> EEGNet:
    """Load trained model."""
    # Load the state dict
    state_dict = torch.load(path, map_location='cpu')
    
    # Create model instance with default parameters
    # Note: These should match the parameters used during training
    model = EEGNet(C=22, T=256)  # Default values, adjust if needed
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _format_float(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3g}"


class LogHandler:
    """Custom log handler to capture logs in GUI."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def write(self, message):
        if message.strip():  # Only write non-empty messages
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
            self.text_widget.update()
    
    def flush(self):
        pass


class App(tk.Tk):
    """Model trainer and explorer app UI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("EEGNet Model Trainer and Explorer")
        self.geometry("1200x800")
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_training_tab()
        self.create_logs_tab()
        self.create_reports_tab()
        self.create_exploration_tab()
        
        # Variables for tracking processes
        self.current_process = None
        self.reports_data = {}
    
    def create_training_tab(self):
        """Create the training pipeline tab."""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training Pipeline")
        
        # Title
        title = ttk.Label(training_frame, text="EEGNet Training Pipeline", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Step 1: Fetch Data
        step1_frame = ttk.LabelFrame(training_frame, text="Step 1: Fetch Data", padding=10)
        step1_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(step1_frame, text="Data Source:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.source_var = tk.StringVar(value="kaggle")
        source_combo = ttk.Combobox(step1_frame, textvariable=self.source_var, values=["kaggle", "moabb"])
        source_combo.grid(row=0, column=1, padx=5)
        
        fetch_btn = ttk.Button(step1_frame, text="Fetch Data", command=self.fetch_data)
        fetch_btn.grid(row=0, column=2, padx=10)
        
        # Step 2: Preprocess Data
        step2_frame = ttk.LabelFrame(training_frame, text="Step 2: Preprocess Data", padding=10)
        step2_frame.pack(fill=tk.X, padx=10, pady=5)
        
        preprocess_btn = ttk.Button(step2_frame, text="Preprocess Data", command=self.preprocess_data)
        preprocess_btn.pack(side=tk.LEFT, padx=5)
        
        # Step 3: Train Model
        step3_frame = ttk.LabelFrame(training_frame, text="Step 3: Train Model", padding=10)
        step3_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(step3_frame, text="Training Type:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.training_type_var = tk.StringVar(value="Within-Subject")
        training_combo = ttk.Combobox(step3_frame, textvariable=self.training_type_var, 
                                    values=["Within-Subject", "Cross-Subject"])
        training_combo.grid(row=0, column=1, padx=5)
        
        self.generate_report_var = tk.BooleanVar(value=True)
        report_check = ttk.Checkbutton(step3_frame, text="Generate Report", variable=self.generate_report_var)
        report_check.grid(row=0, column=2, padx=10)
        
        train_btn = ttk.Button(step3_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=0, column=3, padx=10)
        
        # Progress bar
        self.progress = Progressbar(training_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(training_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
    
    def create_logs_tab(self):
        """Create the logs viewing tab."""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Title
        title = ttk.Label(logs_frame, text="Real-time Logs", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=25, width=120)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Clear logs button
        clear_btn = ttk.Button(logs_frame, text="Clear Logs", command=self.clear_logs)
        clear_btn.pack(pady=5)
    
    def create_reports_tab(self):
        """Create the reports viewing tab."""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="Training Reports")
        
        # Title
        title = ttk.Label(reports_frame, text="Training Results", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Refresh button
        refresh_btn = ttk.Button(reports_frame, text="Refresh Reports", command=self.load_reports)
        refresh_btn.pack(pady=5)
        
        # Create notebook for different report types
        self.reports_notebook = ttk.Notebook(reports_frame)
        self.reports_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load initial reports
        self.load_reports()
    
    def create_exploration_tab(self):
        """Create the model exploration tab."""
        exploration_frame = ttk.Frame(self.notebook)
        self.notebook.add(exploration_frame, text="Model Exploration")
        
        # Title
        title = ttk.Label(exploration_frame, text="Model Filter Visualization", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Model selection
        model_frame = ttk.LabelFrame(exploration_frame, text="Select Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Subject (for Within-Subject):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.subject_var = tk.StringVar(value="01")
        subject_combo = ttk.Combobox(model_frame, textvariable=self.subject_var, 
                                   values=[f"{i:02d}" for i in range(1, 10)])
        subject_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(model_frame, text="Model Type:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.model_type_var = tk.StringVar(value="Within-Subject")
        model_type_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var, 
                                      values=["Within-Subject", "Cross-Subject"])
        model_type_combo.grid(row=0, column=3, padx=5)
        
        # Visualization buttons
        viz_frame = ttk.LabelFrame(exploration_frame, text="Visualizations", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(viz_frame, text="Plot Temporal Filters", 
                  command=self.plot_temporal_filters_gui).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(viz_frame, text="Plot Spatial Filters", 
                  command=self.plot_spatial_filters_gui).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(viz_frame, text="Plot Power Spectra", 
                  command=self.plot_power_spectra_gui).grid(row=0, column=2, padx=5, pady=5)
    
    def fetch_data(self):
        """Execute data fetching in a separate thread."""
        def run_fetch():
            self.status_var.set("Fetching data...")
            self.progress.start()
            try:
                cmd = [sys.executable, "-m", "eegnet_repl.fetch", "--src", self.source_var.get()]
                self.run_subprocess(cmd, "Data fetching completed")
            except Exception as e:
                messagebox.showerror("Error", f"Data fetching failed: {str(e)}")
                self.status_var.set("Error in data fetching")
            finally:
                self.progress.stop()
        
        threading.Thread(target=run_fetch, daemon=True).start()
    
    def preprocess_data(self):
        """Execute data preprocessing in a separate thread."""
        def run_preprocess():
            self.status_var.set("Preprocessing data...")
            self.progress.start()
            try:
                cmd = [sys.executable, "-m", "eegnet_repl.dataset", "--src", self.source_var.get()]
                self.run_subprocess(cmd, "Data preprocessing completed")
            except Exception as e:
                messagebox.showerror("Error", f"Data preprocessing failed: {str(e)}")
                self.status_var.set("Error in data preprocessing")
            finally:
                self.progress.stop()
        
        threading.Thread(target=run_preprocess, daemon=True).start()
    
    def train_model(self):
        """Execute model training in a separate thread."""
        def run_train():
            self.status_var.set("Training model...")
            self.progress.start()
            try:
                cmd = [sys.executable, "-m", "eegnet_repl.train", 
                      "--trainingType", self.training_type_var.get(),
                      "--generateReport", str(self.generate_report_var.get())]
                self.run_subprocess(cmd, "Model training completed")
                # Refresh reports after training
                self.after(1000, self.load_reports)
            except Exception as e:
                messagebox.showerror("Error", f"Model training failed: {str(e)}")
                self.status_var.set("Error in model training")
            finally:
                self.progress.stop()
        
        threading.Thread(target=run_train, daemon=True).start()
    
    def run_subprocess(self, cmd, success_message):
        """Run subprocess and capture output to logs."""
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True)
            self.current_process = process
            
            for line in process.stdout:
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                self.log_text.update()
            
            process.wait()
            if process.returncode == 0:
                self.status_var.set(success_message)
                self.log_text.insert(tk.END, f"\n=== {success_message} ===\n")
            else:
                self.status_var.set("Process failed")
                self.log_text.insert(tk.END, f"\n=== Process failed with return code {process.returncode} ===\n")
                
        except Exception as e:
            self.log_text.insert(tk.END, f"\nError running subprocess: {str(e)}\n")
            raise
    
    def clear_logs(self):
        """Clear the log text area."""
        self.log_text.delete(1.0, tk.END)
    
    def load_reports(self):
        """Load and display training reports."""
        self.reports_data = get_report()
        
        # Clear existing report tabs
        for tab in self.reports_notebook.tabs():
            self.reports_notebook.forget(tab)
        
        if 'within_subject' in self.reports_data:
            self.create_within_subject_report_tab()
        
        if 'cross_subject' in self.reports_data:
            self.create_cross_subject_report_tab()
        
        if not self.reports_data:
            # Show message if no reports found
            no_reports_frame = ttk.Frame(self.reports_notebook)
            self.reports_notebook.add(no_reports_frame, text="No Reports")
            ttk.Label(no_reports_frame, text="No training reports found.\nPlease run training first.", 
                     font=('Arial', 12)).pack(expand=True)
    
    def create_within_subject_report_tab(self):
        """Create tab for within-subject training results."""
        ws_frame = ttk.Frame(self.reports_notebook)
        self.reports_notebook.add(ws_frame, text="Within-Subject")
        
        report = self.reports_data['within_subject']
        
        # Create scrollable frame
        canvas = tk.Canvas(ws_frame)
        scrollbar = ttk.Scrollbar(ws_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Overall results
        overall_frame = ttk.LabelFrame(scrollable_frame, text="Overall Results", padding=10)
        overall_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overall_results = report['overall_results']
        ttk.Label(overall_frame, text=f"Average Test Accuracy: {overall_results['average_test_accuracy']}%", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Best Subject: {overall_results['best_subject_accuracy']}%").pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Worst Subject: {overall_results['worst_subject_accuracy']}%").pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Standard Deviation: {overall_results['accuracy_std']}%").pack(anchor=tk.W)
        
        # Per-subject results table
        table_frame = ttk.LabelFrame(scrollable_frame, text="Per-Subject Results", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for results
        columns = ('Subject', 'Accuracy', 'Rank')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        for result in report['per_subject_results']:
            tree.insert('', tk.END, values=(
                f"Subject {result['subject_id']}",
                f"{result['test_accuracy']}%",
                result['performance_rank']
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Add bar chart
        self.create_accuracy_chart(scrollable_frame, report['per_subject_results'], "Within-Subject")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_cross_subject_report_tab(self):
        """Create tab for cross-subject training results."""
        cs_frame = ttk.Frame(self.reports_notebook)
        self.reports_notebook.add(cs_frame, text="Cross-Subject")
        
        report = self.reports_data['cross_subject']
        
        # Create scrollable frame
        canvas = tk.Canvas(cs_frame)
        scrollbar = ttk.Scrollbar(cs_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Overall results
        overall_frame = ttk.LabelFrame(scrollable_frame, text="Overall Results", padding=10)
        overall_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overall_results = report['overall_results']
        ttk.Label(overall_frame, text=f"Average Test Accuracy: {overall_results['average_test_accuracy']}%", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Standard Error: ±{overall_results['standard_error']}%").pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Best Subject: {overall_results['best_subject_accuracy']}%").pack(anchor=tk.W)
        ttk.Label(overall_frame, text=f"Worst Subject: {overall_results['worst_subject_accuracy']}%").pack(anchor=tk.W)
        
        # Per-subject results table
        table_frame = ttk.LabelFrame(scrollable_frame, text="Per-Subject Results (Test Subject)", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Test Subject', 'Accuracy', 'Rank')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        for result in report['per_subject_results']:
            tree.insert('', tk.END, values=(
                f"Subject {result['test_subject_id']}",
                f"{result['test_accuracy']}%",
                result['performance_rank']
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Add bar chart
        self.create_accuracy_chart(scrollable_frame, report['per_subject_results'], "Cross-Subject")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_accuracy_chart(self, parent, results, title_prefix):
        """Create a bar chart of test accuracies."""
        chart_frame = ttk.LabelFrame(parent, text="Accuracy Comparison", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Extract data
        if 'subject_id' in results[0]:  # Within-subject
            subjects = [f"S{r['subject_id']}" for r in results]
            accuracies = [r['test_accuracy'] for r in results]
        else:  # Cross-subject
            subjects = [f"S{r['test_subject_id']}" for r in results]
            accuracies = [r['test_accuracy'] for r in results]
        
        bars = ax.bar(subjects, accuracies, color='steelblue', alpha=0.7)
        ax.set_xlabel('Subject')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{title_prefix} - Test Accuracy by Subject')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{acc}%', ha='center', va='bottom')
        
        # Add average line
        avg_acc = np.mean(accuracies)
        ax.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_acc:.2f}%')
        ax.legend()
        
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_temporal_filters_gui(self):
        """Plot temporal filters using GUI selection."""
        try:
            model_path = self.get_selected_model_path()
            if model_path and model_path.exists():
                model = load_model(model_path)
                plot_temporal_filters(model.state_dict())
            else:
                messagebox.showerror("Error", "Selected model file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot temporal filters: {str(e)}")
    
    def plot_spatial_filters_gui(self):
        """Plot spatial filters using GUI selection."""
        try:
            model_path = self.get_selected_model_path()
            if model_path and model_path.exists():
                model = load_model(model_path)
                plot_spatial_filters(model.state_dict())
            else:
                messagebox.showerror("Error", "Selected model file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot spatial filters: {str(e)}")
    
    def plot_power_spectra_gui(self):
        """Plot power spectra using GUI selection."""
        try:
            model_path = self.get_selected_model_path()
            if model_path and model_path.exists():
                model = load_model(model_path)
                plot_power_spectra_of_temporal_filters(model.state_dict())
            else:
                messagebox.showerror("Error", "Selected model file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot power spectra: {str(e)}")
    
    def get_selected_model_path(self) -> Path:
        """Get the path to the selected model."""
        paths = Paths.from_here()
        
        if self.model_type_var.get() == "Within-Subject":
            model_filename = f"subject_{self.subject_var.get()}_best_model.pth"
        else:
            model_filename = "cross_subject_best_model.pth"
        
        return paths.models / model_filename

# Complementary functions for plotting

def plot_temporal_filters(model_dict) -> None:
    # Plot learned temporal filters
    temporal_filters = model_dict['temporal.0.weight']
    n_filters = temporal_filters.shape[0]
    plt.figure(figsize=(10, 6))
    for i in range(n_filters):
        plt.plot(temporal_filters[i, 0, 0].cpu().numpy(), label=f'Filter {i+1}')
    plt.title('Learned Temporal Filters')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_spatial_filters(model_dict) -> None:
    # Visualize learned spatial filters using montages
    # Isolate the area that has electrodes and plot topomaps
    info = mne.create_info(
        ch_names=[
                'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5',  'C3',  'C1',  'Cz', 
                'C2',  'C4',  'C6',  'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',  'Pz', 
                'P2',  'POz'
            ],
        sfreq=128,
        ch_types='eeg'
    )
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    spatial_filters = model_dict['spatial.weight']  # shape = (F1*D, F1, C, 1)
    num_filters = spatial_filters.shape[0]
    # Plot in a grid
    n_cols = 4
    n_rows = int(np.ceil(num_filters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i in range(num_filters):
        ax = axes[i // n_cols, i % n_cols]
        filter_weights = spatial_filters[i, 0, :, 0].cpu().numpy()  # shape = (C,)
        plot_topomap(filter_weights, info, axes=ax, show=False, cmap='viridis')
        ax.set_title(f'Spatial Filter {i+1}')
    plt.tight_layout()
    plt.show()

def PS(time_signal, f_sampling, method='ps'):
    fft = np.fft.fft(time_signal)
    mag_squared = np.real(fft * np.conjugate(fft))
    f = np.fft.fftfreq(len(time_signal), 1/f_sampling)

    if method == 'psd':
        scaling_factor = 2 / (f_sampling * len(time_signal))
    else:
        scaling_factor = 2 / (len(time_signal)**2) 

    PS = scaling_factor * mag_squared
    return f, PS

def plot_power_spectra_of_temporal_filters(model_dict) -> None:
    # Plotting learned temporal filters in subplots
    temporal_filters = model_dict['temporal.0.weight']
    n_filters = temporal_filters.shape[0]
    n_cols = 4
    n_rows = n_filters // n_cols + int(n_filters % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    for i in range(n_filters):
        row = i // n_cols
        col = i % n_cols
        #f, ps_version1 = PS(time_signal=temporal_filters[i, 0, 0].cpu().numpy(), f_sampling=128, method='psd') # Power Spectral Density scaling
        f, ps_version2 = PS(time_signal=temporal_filters[i, 0, 0].cpu().numpy(), f_sampling=128, method='ps')  # Power Spectrum scaling
        #axes[row, col].plot(f[0:len(f)//2-1], ps_version1[0:len(f)//2-1], 'ko-')
        axes[row, col].plot(f[0:len(f)//2-1], ps_version2[0:len(f)//2-1], 'ro-')
        axes[row, col].set_title(f'Temporal Filter {i+1}')
        axes[row, col].set_xlabel('Frequency (Hz)')
        axes[row, col].set_ylabel('Power (dB)')
        axes[row, col].set_xticks(range(0,51,10))
        #axes[row, col].legend(['PSD', 'Power Spectrum'])
    plt.tight_layout()
    plt.show()

def get_report() -> dict:
    """Load the most recent training reports."""
    paths = Paths.from_here()
    reports = {}
    
    # Try to load within-subject report
    ws_report_path = paths.reports / "latest_within_subject_report.json"
    if ws_report_path.exists():
        try:
            with open(ws_report_path, 'r', encoding='utf-8') as f:
                reports['within_subject'] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading within-subject report: {e}")
    
    # Try to load cross-subject report
    cs_report_path = paths.reports / "latest_cross_subject_report.json"
    if cs_report_path.exists():
        try:
            with open(cs_report_path, 'r', encoding='utf-8') as f:
                reports['cross_subject'] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading cross-subject report: {e}")
    
    return reports


def main() -> None:
    """Run the UI."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
