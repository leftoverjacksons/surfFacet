import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from datetime import datetime
import os
import re
from pathlib import Path

class CSVTemperatureAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Temperature & Humidity Analyzer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.csv_files = []
        self.current_df = None
        self.comparison_df = None
        self.detected_columns = {}
        self.comparison_detected_columns = {}
        
        # Initialize settings variables with defaults
        self.ma_window_var = tk.IntVar(value=5)
        self.show_raw_var = tk.BooleanVar(value=True)
        self.show_trend_var = tk.BooleanVar(value=True)
        
        # Setup GUI
        self.setup_gui()
        self.scan_for_csv_files()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Plot area row
        
        # File selection section
        ttk.Label(main_frame, text="Select CSV File:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0,5))
        
        self.csv_var = tk.StringVar()
        self.csv_dropdown = ttk.Combobox(main_frame, textvariable=self.csv_var, width=50, state="readonly")
        self.csv_dropdown.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0,5))
        self.csv_dropdown.bind('<<ComboboxSelected>>', self.on_csv_selected)
        
        # Comparison file selection
        ttk.Label(main_frame, text="Compare with:", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, pady=(5,5))
        
        self.comparison_var = tk.StringVar()
        self.comparison_dropdown = ttk.Combobox(main_frame, textvariable=self.comparison_var, width=50, state="readonly")
        self.comparison_dropdown.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(5,5))
        self.comparison_dropdown.bind('<<ComboboxSelected>>', self.on_comparison_selected)
        
        # Time shift control
        shift_frame = ttk.Frame(main_frame)
        shift_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5,10))
        
        ttk.Label(shift_frame, text="Time shift comparison:").pack(side=tk.LEFT, padx=(0,10))
        
        self.time_shift_var = tk.DoubleVar(value=0.0)
        self.time_shift_scale = ttk.Scale(shift_frame, from_=-120, to=120, 
                                         variable=self.time_shift_var, 
                                         orient=tk.HORIZONTAL, length=200,
                                         command=self.on_time_shift_changed)
        self.time_shift_scale.pack(side=tk.LEFT, padx=(0,10))
        
        self.shift_label = ttk.Label(shift_frame, text="0.0 min")
        self.shift_label.pack(side=tk.LEFT, padx=(5,0))
        
        ttk.Button(shift_frame, text="Reset", command=self.reset_time_shift).pack(side=tk.LEFT, padx=(10,0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(buttons_frame, text="Browse Files", command=self.browse_files).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(buttons_frame, text="Refresh", command=self.scan_for_csv_files).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(buttons_frame, text="Analyze", command=self.analyze_data).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(buttons_frame, text="Export Plots", command=self.export_plots).pack(side=tk.LEFT, padx=(0,10))
        
        # Column detection info
        self.info_frame = ttk.LabelFrame(main_frame, text="Detected Columns", padding="5")
        self.info_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0,10))
        
        self.info_text = tk.Text(self.info_frame, height=4, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_frame.columnconfigure(0, weight=1)
        
        # Plot area with notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a CSV file to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,0))
        
    def scan_for_csv_files(self):
        """Scan current directory and common subdirectories for CSV files"""
        csv_files = []
        
        # Current directory
        current_dir = Path(".")
        csv_files.extend(list(current_dir.glob("*.csv")))
        
        # Common data directories
        data_dirs = ["data", "csv", "files", "temp_data", "test_data"]
        for dir_name in data_dirs:
            data_path = current_dir / dir_name
            if data_path.exists():
                csv_files.extend(list(data_path.glob("*.csv")))
        
        # Update both dropdowns
        self.csv_files = [str(f) for f in csv_files]
        self.csv_dropdown['values'] = self.csv_files
        self.comparison_dropdown['values'] = ["(None)"] + self.csv_files
        
        if self.csv_files:
            self.status_var.set(f"Found {len(self.csv_files)} CSV files")
        else:
            self.status_var.set("No CSV files found - use Browse to select files")
            
    def browse_files(self):
        """Open file dialog to select CSV files"""
        filenames = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filenames:
            # Add new files to the list
            for filename in filenames:
                if filename not in self.csv_files:
                    self.csv_files.append(filename)
            
            # Update both dropdowns
            self.csv_dropdown['values'] = self.csv_files
            self.comparison_dropdown['values'] = ["(None)"] + self.csv_files
            self.status_var.set(f"Added {len(filenames)} files. Total: {len(self.csv_files)} CSV files available")
            
    def on_comparison_selected(self, event=None):
        """Handle comparison CSV file selection"""
        selected_file = self.comparison_var.get()
        if not selected_file or selected_file == "(None)":
            self.comparison_df = None
            self.comparison_detected_columns = {}
            self.update_column_info()  # Refresh display
            self.status_var.set("Comparison file cleared")
            return
            
        try:
            self.status_var.set("Loading comparison CSV file...")
            self.root.update()
            
            # Load and process the comparison CSV
            df = self.load_and_process_csv(selected_file)
            
            # Detect column types
            self.comparison_detected_columns = self.detect_column_types(df)
            
            # Process time column
            df, time_col = self.process_time_column(df, self.comparison_detected_columns['time'])
            
            self.comparison_df = df
            
            # Update info display to show both datasets
            self.update_column_info()
            
            self.status_var.set(f"Loaded comparison: {len(df)} rows from {os.path.basename(selected_file)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load comparison CSV file:\n{str(e)}")
            self.status_var.set("Error loading comparison file")
            
    def on_time_shift_changed(self, value):
        """Handle time shift slider changes"""
        shift_minutes = float(value)
        self.shift_label.config(text=f"{shift_minutes:.1f} min")
        
        # Update the column info to show current shift
        if self.comparison_df is not None:
            self.update_column_info()
        
        # Auto-refresh plots when time shift changes if both datasets are loaded
        if self.current_df is not None and self.comparison_df is not None:
            # Add a small delay to avoid too many refreshes while dragging
            if hasattr(self, '_shift_timer'):
                self.root.after_cancel(self._shift_timer)
            self._shift_timer = self.root.after(200, self.analyze_data)  # 200ms delay
            
    def reset_time_shift(self):
        """Reset time shift to zero"""
        self.time_shift_var.set(0.0)
        self.shift_label.config(text="0.0 min")
        if self.comparison_df is not None:
            self.update_column_info()
            # Auto-refresh plot when reset
            if self.current_df is not None:
                self.analyze_data()
            
    def detect_column_types(self, df):
        """Simple column detection - if it's time, mark it. If it's numeric with good data, plot it."""
        detected = {
            'time': [],
            'data': [],  # Just "data" - we'll style it later
            'humidity': [],  # Separate humidity columns
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Time detection - simple keyword check
            if 'time' in col_lower:
                detected['time'].append(col)
            # Skip V3 Raw columns - we don't want those (be more aggressive about this)
            elif 'v3' in col_lower and ('raw' in col_lower or col_lower.endswith(' raw')):
                detected['other'].append(col)
            # Humidity detection - look for RH or humidity keywords
            elif 'rh' in col_lower or 'humidity' in col_lower:
                # Try to convert to numeric
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    completeness = numeric_data.notna().sum() / len(df)
                    
                    # If it's mostly numeric data (>50% complete), plot it as humidity
                    if completeness > 0.5:
                        detected['humidity'].append(col)
                    else:
                        detected['other'].append(col)
                except:
                    detected['other'].append(col)
            else:
                # Try to convert to numeric
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    completeness = numeric_data.notna().sum() / len(df)
                    
                    # If it's mostly numeric data (>50% complete), plot it
                    if completeness > 0.5:
                        # Double-check: still skip any V3 Raw that slipped through
                        if 'v3' in col_lower and 'raw' in col_lower:
                            detected['other'].append(col)
                        else:
                            detected['data'].append(col)
                    else:
                        detected['other'].append(col)
                except:
                    detected['other'].append(col)
                    
        return detected
    
    def get_sensor_style(self, column_name):
        """Get standardized color and style for sensor based on brand and type"""
        col_lower = column_name.lower()
        
        # Determine brand and type
        is_globe = any(keyword in col_lower for keyword in ['globe', 'tg'])
        
        # Brand-based colors
        if 'tsi' in col_lower:
            color = '#dc2626'  # Red
        elif 'kestrel' in col_lower:
            color = '#2563eb'  # Blue  
        elif 'v2' in col_lower:
            color = '#000000'  # Black
        elif 'v3' in col_lower:
            color = '#ea580c'  # Orange
        else:
            color = '#6b7280'  # Gray for unknown
            
        # Style based on reading type
        linestyle = '--' if is_globe else '-'
        linewidth = 3 if 'v3' in col_lower else 2
        
        # Generate label from column name
        label = column_name
        
        return {
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'label': label
        }
        
    def load_and_process_csv(self, filepath):
        """Load and process the selected CSV file"""
        try:
            # Try to read the file, handling various formats
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file appears to be empty")
            
            # Handle files that might have metadata at the top
            if df.shape[1] < 2:
                # Try to find where the actual CSV data starts
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                
                csv_start_line = 0
                for i, line in enumerate(lines):
                    if ',' in line and not line.strip().startswith('#'):
                        # Count commas to see if this looks like a data row
                        comma_count = line.count(',')
                        if comma_count >= 2:  # At least 3 columns
                            csv_start_line = i
                            break
                
                if csv_start_line > 0:
                    df = pd.read_csv(filepath, skiprows=csv_start_line)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Remove completely empty rows
            df = df.dropna(axis=0, how='all')
            
            if df.empty:
                raise ValueError("No valid data found in the CSV file")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
            
    def process_time_column(self, df, time_cols):
        """Process time column(s) to create a datetime index"""
        if not time_cols:
            # Create a simple index if no time column
            df['index'] = range(len(df))
            return df, 'index'
            
        time_col = time_cols[0]  # Use first detected time column
        
        try:
            # Only try simple time formats to avoid invalid dates
            time_formats = ['%H:%M', '%H:%M:%S']
            
            for fmt in time_formats:
                try:
                    df['datetime'] = pd.to_datetime(df[time_col], format=fmt)
                    df = df.sort_values('datetime').reset_index(drop=True)
                    return df, 'datetime'
                except:
                    continue
                    
            # If time parsing fails, fall back to index
            print("Time parsing failed, using simple index")
            df['index'] = range(len(df))
            return df, 'index'
                
        except Exception:
            # Fall back to simple index
            df['index'] = range(len(df))
            return df, 'index'
            
    def on_csv_selected(self, event=None):
        """Handle CSV file selection"""
        selected_file = self.csv_var.get()
        if not selected_file:
            return
            
        try:
            self.status_var.set("Loading CSV file...")
            self.root.update()
            
            # Load and process the CSV
            df = self.load_and_process_csv(selected_file)
            
            # Detect column types
            self.detected_columns = self.detect_column_types(df)
            
            # Process time column
            df, time_col = self.process_time_column(df, self.detected_columns['time'])
            
            self.current_df = df
            
            # Update info display
            self.update_column_info()
            
            self.status_var.set(f"Loaded {len(df)} rows from {os.path.basename(selected_file)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
            self.status_var.set("Error loading file")
            
    def update_column_info(self):
        """Update the column information display"""
        self.info_text.delete(1.0, tk.END)
        
        if not self.detected_columns:
            self.info_text.insert(tk.END, "No file selected")
            return
            
        info_text = "MAIN DATASET:\n"
        for col_type, cols in self.detected_columns.items():
            if cols:
                info_text += f"{col_type.title()}: {', '.join(cols)}\n"
                
        if self.current_df is not None:
            info_text += f"Data shape: {self.current_df.shape[0]} rows × {self.current_df.shape[1]} columns\n"
            
        # Add comparison info if loaded
        if self.comparison_df is not None and self.comparison_detected_columns:
            info_text += "\nCOMPARISON DATASET:\n"
            for col_type, cols in self.comparison_detected_columns.items():
                if cols:
                    info_text += f"{col_type.title()}: {', '.join(cols)}\n"
            info_text += f"Data shape: {self.comparison_df.shape[0]} rows × {self.comparison_df.shape[1]} columns\n"
            
            # Show time shift info
            shift_value = self.time_shift_var.get()
            if shift_value != 0:
                info_text += f"Time shift: {shift_value:.1f} minutes\n"
            
        self.info_text.insert(tk.END, info_text)
        
    def calculate_moving_average(self, series, window=None):
        """Calculate moving average for smoothing"""
        if window is None:
            window = self.ma_window_var.get()
        return series.rolling(window=window, center=True, min_periods=1).mean()
        
    def create_data_plot(self):
        """Create data analysis plot - plots all numeric columns including comparison"""
        if self.current_df is None or not self.detected_columns['data']:
            return None
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Determine x-axis (time or index)
        if 'datetime' in self.current_df.columns:
            x_data = self.current_df['datetime']
            x_label = 'Time'
            use_time = True
        elif 'index' in self.current_df.columns:
            x_data = self.current_df['index']
            x_label = 'Measurement #'
            use_time = False
        else:
            x_data = range(len(self.current_df))
            x_label = 'Index'
            use_time = False
            
        # Plot main dataset
        for col in self.detected_columns['data']:
            if col in self.current_df.columns:
                # Convert to numeric
                data = pd.to_numeric(self.current_df[col], errors='coerce')
                if data.notna().sum() > 0:
                    style = self.get_sensor_style(col)
                    
                    # Raw data points (if enabled)
                    if self.show_raw_var.get():
                        ax.scatter(x_data, data, 
                                 marker='x', s=25, color=style['color'], alpha=0.6,
                                 label=f"{style['label']} (Raw)")
                    
                    # Smoothed trend (if enabled)
                    if self.show_trend_var.get():
                        smoothed = self.calculate_moving_average(data)
                        ax.plot(x_data, smoothed, 
                               color=style['color'], 
                               linestyle=style['linestyle'],
                               linewidth=style['linewidth'],
                               label=f"{style['label']} (Trend)")
        
        # Plot comparison dataset if loaded
        if self.comparison_df is not None and self.comparison_detected_columns.get('data'):
            self.plot_comparison_data(ax, use_time)
        
        # Safe time axis formatting - only if we actually have time data
        if use_time and 'datetime' in self.current_df.columns:
            try:
                import matplotlib.dates as mdates
                # Use only hourly locators to avoid tick overflow
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            except:
                # If any time formatting fails, just skip it
                pass
        
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°F)', fontsize=12, fontweight='bold')
        
        # Update title to show comparison status
        title = 'Temperature Analysis'
        if self.comparison_df is not None:
            shift = self.time_shift_var.get()
            if shift != 0:
                title += f' (Comparison shifted {shift:+.1f} min)'
            else:
                title += ' (with Comparison)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Add simple hover functionality
        self.add_hover_functionality(fig, ax, x_data, use_time)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_data(self, ax, use_time):
        """Plot the comparison dataset on the same axes"""
        # Get time shift in minutes
        time_shift_minutes = self.time_shift_var.get()
        
        # Determine x-axis for comparison data
        if 'datetime' in self.comparison_df.columns:
            # Apply time shift
            x_data_comp = self.comparison_df['datetime'] + pd.Timedelta(minutes=time_shift_minutes)
        elif 'index' in self.comparison_df.columns:
            # For index-based data, convert shift to data points (assume 5-min intervals)
            shift_points = int(time_shift_minutes / 5) if time_shift_minutes != 0 else 0
            x_data_comp = self.comparison_df['index'] + shift_points
        else:
            x_data_comp = range(len(self.comparison_df))
            
        # Plot comparison dataset with modified styling - ALL AT 50% OPACITY
        for col in self.comparison_detected_columns['data']:
            if col in self.comparison_df.columns:
                # Convert to numeric
                data = pd.to_numeric(self.comparison_df[col], errors='coerce')
                if data.notna().sum() > 0:
                    style = self.get_sensor_style(col)
                    
                    # Modify style for comparison (50% transparent, dashed)
                    comp_style = style.copy()
                    comp_style['linestyle'] = '--' if style['linestyle'] == '-' else ':'
                    
                    # Raw data points (if enabled) - 50% opacity
                    if self.show_raw_var.get():
                        ax.scatter(x_data_comp, data, 
                                 marker='o', s=15, color=style['color'], alpha=0.5,
                                 label=f"{style['label']} Comp (Raw)")
                    
                    # Smoothed trend (if enabled) - 50% opacity
                    if self.show_trend_var.get():
                        smoothed = self.calculate_moving_average(data)
                        ax.plot(x_data_comp, smoothed, 
                               color=style['color'], 
                               linestyle=comp_style['linestyle'],
                               linewidth=style['linewidth'],
                               alpha=0.5,
                               label=f"{style['label']} Comp (Trend)")
    
    def add_hover_functionality(self, fig, ax, x_data, use_time):
        """Add simple mouseover functionality with line highlighting"""
        # Create annotation box (initially invisible)
        annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        
        # Store original line properties for highlighting
        self.original_line_props = {}
        for line in ax.get_lines():
            self.original_line_props[line] = {
                'linewidth': line.get_linewidth(),
                'alpha': line.get_alpha() or 1.0
            }
        
        self.highlighted_line = None
        
        def on_hover(event):
            if event.inaxes == ax:
                # Reset any previously highlighted line
                if self.highlighted_line:
                    props = self.original_line_props[self.highlighted_line]
                    self.highlighted_line.set_linewidth(props['linewidth'])
                    self.highlighted_line.set_alpha(props['alpha'])
                    self.highlighted_line = None
                
                # Check if we're hovering over any data points
                for line in ax.get_lines():
                    contains, info = line.contains(event)
                    if contains:
                        # Highlight the current line
                        line.set_linewidth(self.original_line_props[line]['linewidth'] + 2)
                        line.set_alpha(1.0)
                        self.highlighted_line = line
                        
                        # Get the data point index
                        ind = info['ind'][0]
                        
                        # Get x and y coordinates
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        
                        if ind < len(xdata) and ind < len(ydata):
                            x_val = xdata[ind]
                            y_val = ydata[ind]
                            
                            # Get the actual time value from the dataset
                            time_display = self.get_time_value_at_index(ind, use_time)
                            
                            # Format the display text
                            text = f"{line.get_label()}\n{time_display}\nValue: {y_val:.1f}°F"
                            
                            # Special V3 comparison for temperature data
                            if 'v3' in line.get_label().lower():
                                v3_comparison = self.get_v3_temperature_comparison(ind, y_val)
                                if v3_comparison:
                                    text += v3_comparison
                            
                            # Update annotation
                            annot.xy = (x_val, y_val)
                            annot.set_text(text)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return
                
                # If not hovering over any line, hide annotation
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def get_time_value_at_index(self, index, use_time):
        """Get the actual time value from the dataset at the given index"""
        if not hasattr(self, 'current_df') or index >= len(self.current_df):
            return f"Point: {index}"
            
        # Try to get time from datetime column first
        if use_time and 'datetime' in self.current_df.columns:
            try:
                time_val = self.current_df.iloc[index]['datetime']
                if pd.notna(time_val):
                    return f"Time: {time_val.strftime('%H:%M:%S')}"
            except:
                pass
        
        # Try to get time from original time column
        time_cols = [col for col in self.current_df.columns if 'time' in col.lower()]
        if time_cols:
            try:
                time_val = self.current_df.iloc[index][time_cols[0]]
                if pd.notna(time_val):
                    return f"Time: {time_val}"
            except:
                pass
                
        # Fallback to point number
        return f"Point: {index}"
    
    def get_v3_temperature_comparison(self, index, v3_value):
        """Get V3 comparison with TSI and Kestrel for temperature"""
        if not hasattr(self, 'current_df') or index >= len(self.current_df):
            return ""
            
        comparison_text = ""
        
        # Look for TSI ambient temperature
        tsi_cols = [col for col in self.current_df.columns if 'tsi' in col.lower() and 'amb' in col.lower()]
        if tsi_cols:
            tsi_col = tsi_cols[0]
            tsi_val = pd.to_numeric(self.current_df.iloc[index][tsi_col], errors='coerce')
            if pd.notna(tsi_val):
                diff_degrees = v3_value - tsi_val
                diff_percent = (diff_degrees / tsi_val) * 100 if tsi_val != 0 else 0
                comparison_text += f"\nvs TSI: {diff_degrees:+.1f}°F ({diff_percent:+.1f}%)"
        
        # Look for Kestrel ambient temperature  
        kestrel_cols = [col for col in self.current_df.columns if 'kestrel' in col.lower() and 'amb' in col.lower()]
        if kestrel_cols:
            kestrel_col = kestrel_cols[0]
            kestrel_val = pd.to_numeric(self.current_df.iloc[index][kestrel_col], errors='coerce')
            if pd.notna(kestrel_val):
                diff_degrees = v3_value - kestrel_val
                diff_percent = (diff_degrees / kestrel_val) * 100 if kestrel_val != 0 else 0
                comparison_text += f"\nvs Kestrel: {diff_degrees:+.1f}°F ({diff_percent:+.1f}%)"
                
        return comparison_text
    
    def create_humidity_plot(self):
        """Create humidity analysis plot including comparison"""
        if self.current_df is None or not self.detected_columns['humidity']:
            return None
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Determine x-axis (time or index)
        if 'datetime' in self.current_df.columns:
            x_data = self.current_df['datetime']
            x_label = 'Time'
            use_time = True
        elif 'index' in self.current_df.columns:
            x_data = self.current_df['index']
            x_label = 'Measurement #'
            use_time = False
        else:
            x_data = range(len(self.current_df))
            x_label = 'Index'
            use_time = False
            
        # Plot main dataset humidity
        for col in self.detected_columns['humidity']:
            if col in self.current_df.columns:
                # Convert to numeric
                data = pd.to_numeric(self.current_df[col], errors='coerce')
                if data.notna().sum() > 0:
                    style = self.get_sensor_style(col)
                    
                    # Raw data points (if enabled)
                    if self.show_raw_var.get():
                        ax.scatter(x_data, data, 
                                 marker='x', s=25, color=style['color'], alpha=0.6,
                                 label=f"{style['label']} (Raw)")
                    
                    # Smoothed trend (if enabled)
                    if self.show_trend_var.get():
                        smoothed = self.calculate_moving_average(data)
                        ax.plot(x_data, smoothed, 
                               color=style['color'], 
                               linestyle=style['linestyle'],
                               linewidth=style['linewidth'],
                               label=f"{style['label']} (Trend)")
        
        # Plot comparison humidity data if loaded
        if self.comparison_df is not None and self.comparison_detected_columns.get('humidity'):
            self.plot_comparison_humidity(ax, use_time)
        
        # Safe time axis formatting
        if use_time and 'datetime' in self.current_df.columns:
            try:
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            except:
                pass
        
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Humidity (%)', fontsize=12, fontweight='bold')
        
        # Update title to show comparison status
        title = 'Relative Humidity Analysis'
        if self.comparison_df is not None:
            shift = self.time_shift_var.get()
            if shift != 0:
                title += f' (Comparison shifted {shift:+.1f} min)'
            else:
                title += ' (with Comparison)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Add hover functionality for humidity
        self.add_hover_functionality_humidity(fig, ax, x_data, use_time)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_humidity(self, ax, use_time):
        """Plot the comparison humidity dataset on the same axes"""
        # Get time shift in minutes
        time_shift_minutes = self.time_shift_var.get()
        
        # Determine x-axis for comparison data
        if 'datetime' in self.comparison_df.columns:
            # Apply time shift
            x_data_comp = self.comparison_df['datetime'] + pd.Timedelta(minutes=time_shift_minutes)
        elif 'index' in self.comparison_df.columns:
            # For index-based data, convert shift to data points (assume 5-min intervals)
            shift_points = int(time_shift_minutes / 5) if time_shift_minutes != 0 else 0
            x_data_comp = self.comparison_df['index'] + shift_points
        else:
            x_data_comp = range(len(self.comparison_df))
            
        # Plot comparison humidity with modified styling - ALL AT 50% OPACITY
        for col in self.comparison_detected_columns['humidity']:
            if col in self.comparison_df.columns:
                # Convert to numeric
                data = pd.to_numeric(self.comparison_df[col], errors='coerce')
                if data.notna().sum() > 0:
                    style = self.get_sensor_style(col)
                    
                    # Modify style for comparison (50% transparent, dashed)
                    comp_style = style.copy()
                    comp_style['linestyle'] = '--' if style['linestyle'] == '-' else ':'
                    
                    # Raw data points (if enabled) - 50% opacity
                    if self.show_raw_var.get():
                        ax.scatter(x_data_comp, data, 
                                 marker='o', s=15, color=style['color'], alpha=0.5,
                                 label=f"{style['label']} Comp (Raw)")
                    
                    # Smoothed trend (if enabled) - 50% opacity
                    if self.show_trend_var.get():
                        smoothed = self.calculate_moving_average(data)
                        ax.plot(x_data_comp, smoothed, 
                               color=style['color'], 
                               linestyle=comp_style['linestyle'],
                               linewidth=style['linewidth'],
                               alpha=0.5,
                               label=f"{style['label']} Comp (Trend)")
    
    def add_hover_functionality_humidity(self, fig, ax, x_data, use_time):
        """Add mouseover functionality to humidity plot with line highlighting"""
        # Create annotation box (initially invisible)
        annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        
        # Store original line properties for highlighting
        self.original_line_props_humidity = {}
        for line in ax.get_lines():
            self.original_line_props_humidity[line] = {
                'linewidth': line.get_linewidth(),
                'alpha': line.get_alpha() or 1.0
            }
        
        self.highlighted_line_humidity = None
        
        def on_hover(event):
            if event.inaxes == ax:
                # Reset any previously highlighted line
                if self.highlighted_line_humidity:
                    props = self.original_line_props_humidity[self.highlighted_line_humidity]
                    self.highlighted_line_humidity.set_linewidth(props['linewidth'])
                    self.highlighted_line_humidity.set_alpha(props['alpha'])
                    self.highlighted_line_humidity = None
                
                # Check if we're hovering over any data points
                for line in ax.get_lines():
                    contains, info = line.contains(event)
                    if contains:
                        # Highlight the current line
                        line.set_linewidth(self.original_line_props_humidity[line]['linewidth'] + 2)
                        line.set_alpha(1.0)
                        self.highlighted_line_humidity = line
                        
                        # Get the data point index
                        ind = info['ind'][0]
                        
                        # Get x and y coordinates
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        
                        if ind < len(xdata) and ind < len(ydata):
                            x_val = xdata[ind]
                            y_val = ydata[ind]
                            
                            # Get the actual time value from the dataset
                            time_display = self.get_time_value_at_index(ind, use_time)
                            
                            # Format the display text
                            text = f"{line.get_label()}\n{time_display}\nHumidity: {y_val:.1f}%"
                            
                            # Special V3 comparison for humidity data
                            if 'v3' in line.get_label().lower():
                                v3_comparison = self.get_v3_humidity_comparison(ind, y_val)
                                if v3_comparison:
                                    text += v3_comparison
                            
                            # Update annotation
                            annot.xy = (x_val, y_val)
                            annot.set_text(text)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return
                
                # If not hovering over any line, hide annotation
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def get_v3_humidity_comparison(self, index, v3_value):
        """Get V3 comparison with TSI and Kestrel for humidity"""
        if not hasattr(self, 'current_df') or index >= len(self.current_df):
            return ""
            
        comparison_text = ""
        
        # Look for TSI humidity
        tsi_cols = [col for col in self.current_df.columns if 'tsi' in col.lower() and ('rh' in col.lower() or 'humidity' in col.lower())]
        if tsi_cols:
            tsi_col = tsi_cols[0]
            tsi_val = pd.to_numeric(self.current_df.iloc[index][tsi_col], errors='coerce')
            if pd.notna(tsi_val):
                diff_percent_abs = v3_value - tsi_val
                diff_percent_rel = (diff_percent_abs / tsi_val) * 100 if tsi_val != 0 else 0
                comparison_text += f"\nvs TSI: {diff_percent_abs:+.1f}% ({diff_percent_rel:+.1f}%)"
        
        # Look for Kestrel humidity
        kestrel_cols = [col for col in self.current_df.columns if 'kestrel' in col.lower() and ('rh' in col.lower() or 'humidity' in col.lower())]
        if kestrel_cols:
            kestrel_col = kestrel_cols[0]
            kestrel_val = pd.to_numeric(self.current_df.iloc[index][kestrel_col], errors='coerce')
            if pd.notna(kestrel_val):
                diff_percent_abs = v3_value - kestrel_val
                diff_percent_rel = (diff_percent_abs / kestrel_val) * 100 if kestrel_val != 0 else 0
                comparison_text += f"\nvs Kestrel: {diff_percent_abs:+.1f}% ({diff_percent_rel:+.1f}%)"
                
        return comparison_text
    
    def create_options_frame(self):
        """Create an options/settings frame"""
        options_frame = ttk.Frame(self.notebook)
        
        # Main container with padding
        container = ttk.Frame(options_frame, padding="20")
        container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(container, text="Analysis Options & Settings", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Styling options
        style_frame = ttk.LabelFrame(container, text="Plot Styling", padding="10")
        style_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Moving average window
        ttk.Label(style_frame, text="Moving Average Window:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ma_spinbox = ttk.Spinbox(style_frame, from_=1, to=20, textvariable=self.ma_window_var, width=10)
        ma_spinbox.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(style_frame, text="data points").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Show raw data points
        raw_check = ttk.Checkbutton(style_frame, text="Show raw data points (X markers)", 
                                   variable=self.show_raw_var)
        raw_check.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Show trend lines
        trend_check = ttk.Checkbutton(style_frame, text="Show smoothed trend lines", 
                                     variable=self.show_trend_var)
        trend_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Color legend
        color_frame = ttk.LabelFrame(container, text="Sensor Color Legend", padding="10")
        color_frame.pack(fill=tk.X, pady=(0, 15))
        
        legend_text = """TSI Sensors: Red (solid for ambient, dashed for globe)
Kestrel Sensors: Blue (solid for ambient, dashed for globe)  
V2 Sensors: Black (solid line)
V3 Sensors: Orange (thick solid line for ambient, thick dashed for globe)"""
        
        legend_label = ttk.Label(color_frame, text=legend_text, 
                                font=('Courier', 10), justify=tk.LEFT)
        legend_label.pack(anchor=tk.W)
        
        # Data info
        if self.current_df is not None:
            info_frame = ttk.LabelFrame(container, text="Current Dataset Info", padding="10")
            info_frame.pack(fill=tk.X, pady=(0, 15))
            
            # Data statistics
            info_text = f"""File: {os.path.basename(self.csv_var.get()) if self.csv_var.get() else 'None'}
Rows: {len(self.current_df):,}
Columns: {len(self.current_df.columns)}
Data Columns: {len(self.detected_columns.get('data', []))}
Humidity Columns: {len(self.detected_columns.get('humidity', []))}"""
            
            if 'time' in self.current_df.columns and len(self.current_df) > 0:
                info_text += f"\nTime Range: {self.current_df['time'].iloc[0]} to {self.current_df['time'].iloc[-1]}"
            
            info_label = ttk.Label(info_frame, text=info_text, 
                                  font=('Courier', 10), justify=tk.LEFT)
            info_label.pack(anchor=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(button_frame, text="Apply Settings & Refresh Plots", 
                  command=self.apply_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_settings).pack(side=tk.LEFT)
        
        return options_frame
    
    def apply_settings(self):
        """Apply the current settings and refresh plots"""
        self.analyze_data()
        
    def reset_settings(self):
        """Reset all settings to defaults"""
        self.ma_window_var.set(5)
        self.show_raw_var.set(True)
        self.show_trend_var.set(True)
        messagebox.showinfo("Settings Reset", "All settings have been reset to defaults.")
        
    def analyze_data(self):
        """Analyze the selected CSV data and create plots"""
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please select a CSV file first")
            return
            
        try:
            self.status_var.set("Generating analysis plots...")
            self.root.update()
            
            # Clear existing tabs
            for tab in self.notebook.tabs():
                self.notebook.forget(tab)
                
            plots_created = 0
            
            # Create data plot (temperature/main data)
            data_fig = self.create_data_plot()
            if data_fig:
                # Create frame for plot
                plot_frame = ttk.Frame(self.notebook)
                self.notebook.add(plot_frame, text="Temperature")
                
                # Create canvas and toolbar
                canvas = FigureCanvasTkAgg(data_fig, plot_frame)
                canvas.draw()
                
                # Navigation toolbar
                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()
                
                # Pack widgets
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                plots_created += 1
            
            # Create humidity plot if humidity data exists
            humidity_fig = self.create_humidity_plot()
            if humidity_fig:
                # Create frame for humidity plot
                humidity_frame = ttk.Frame(self.notebook)
                self.notebook.add(humidity_frame, text="Humidity")
                
                # Create canvas and toolbar
                canvas = FigureCanvasTkAgg(humidity_fig, humidity_frame)
                canvas.draw()
                
                # Navigation toolbar
                toolbar = NavigationToolbar2Tk(canvas, humidity_frame)
                toolbar.update()
                
                # Pack widgets
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                plots_created += 1
            
            # Add Options tab at the end
            options_frame = self.create_options_frame()
            self.notebook.add(options_frame, text="Options")
                
            if plots_created > 0:
                self.status_var.set(f"Analysis complete - {plots_created} plots generated")
                # Select first data tab by default
                self.notebook.select(0)
            else:
                self.status_var.set("No plots generated - check your data format")
                messagebox.showinfo("Info", "No suitable numeric data found for plotting.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze data:\n{str(e)}")
            self.status_var.set("Analysis failed")
            
    def export_plots(self):
        """Export all current plots as PNG files"""
        if not self.notebook.tabs():
            messagebox.showwarning("Warning", "No plots to export. Run analysis first.")
            return
            
        try:
            # Ask for directory
            directory = filedialog.askdirectory(title="Select directory to save plots")
            if not directory:
                return
                
            # Get base filename from current CSV
            csv_name = os.path.splitext(os.path.basename(self.csv_var.get()))[0]
            
            exported_count = 0
            for tab_id in self.notebook.tabs():
                tab_name = self.notebook.tab(tab_id, "text")
                
                # Skip options tab
                if tab_name == "Options":
                    continue
                
                # Get the figure from the tab
                plot_frame = self.notebook.nametowidget(tab_id)
                for widget in plot_frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        filename = os.path.join(directory, f"{csv_name}_{tab_name.lower()}_plot.png")
                        widget.figure.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                        exported_count += 1
                        break
                        
            messagebox.showinfo("Success", f"Exported {exported_count} plots to {directory}")
            self.status_var.set(f"Exported {exported_count} plots")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plots:\n{str(e)}")

def main():
    root = tk.Tk()
    app = CSVTemperatureAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()