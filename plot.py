import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data(csv_file_path):
    """Load and process the BEACON temperature data"""
    # Read the file and find where CSV data starts
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line that starts with 'time,'
    csv_start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('time,'):
            csv_start_line = i
            break
    
    # Read CSV starting from the correct line
    df = pd.read_csv(csv_file_path, skiprows=csv_start_line)
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # Convert time to datetime for proper sorting
    df['datetime'] = pd.to_datetime(df['time'], format='%H:%M')
    df = df.sort_values('datetime')
    
    # Reset index for proper time series
    df = df.reset_index(drop=True)
    
    # Handle 'NA' values in humidity columns by converting to NaN
    humidity_cols = ['RH TSI', 'RH Kestrel', 'RH V2', 'RH V3']
    for col in humidity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply correction offset to V3 ambient temperature from 13:12 onward
    if 'TA V3' in df.columns:
        # Find the index where time is 13:12 or later
        correction_start_idx = None
        for i, time_val in enumerate(df['time']):
            if time_val >= '13:12':
                correction_start_idx = i
                break
        
        if correction_start_idx is not None:
            # Apply +1.3°F correction to TA V3 from 13:12 onward
            df.loc[correction_start_idx:, 'TA V3'] = df.loc[correction_start_idx:, 'TA V3'] + 1
            print(f"Applied +1°F correction to TA V3 from time {df.loc[correction_start_idx, 'time']} onward ({len(df) - correction_start_idx} data points)")
    
    return df

def calculate_moving_average(series, window=5):
    """Calculate 5-point moving average"""
    return series.rolling(window=window, center=True, min_periods=1).mean()

def create_temperature_visualization(df):
    """Create the main temperature visualization with separate windows"""
    
    # Create separate figures for each plot type
    figures = []
    
    # Define sensor configurations for ambient temperatures
    ambient_config = {
        'TA TSI': {'color': '#dc2626', 'linestyle': '-', 'linewidth': 2, 'label': 'TSI Ambient'},
        'TA Kestrel': {'color': '#2563eb', 'linestyle': '-', 'linewidth': 2, 'label': 'Kestrel Ambient'},
        'TA V2': {'color': '#000000', 'linestyle': '-', 'linewidth': 3, 'label': 'V2 Ambient'},
        'TA V3': {'color': '#ea580c', 'linestyle': '-', 'linewidth': 3, 'label': 'V3 Ambient'}
    }
    
    # Define sensor configurations for globe temperatures
    globe_config = {
        'TG TSI': {'color': '#dc2626', 'linestyle': '--', 'linewidth': 2, 'label': 'TSI Globe'},
        'TG Kestrel': {'color': '#2563eb', 'linestyle': '--', 'linewidth': 2, 'label': 'Kestrel Globe'}
    }
    
    # Define sensor configurations for humidity
    humidity_config = {
        'RH TSI': {'color': '#16a34a', 'linestyle': '-', 'linewidth': 2, 'label': 'TSI Humidity'},
        'RH Kestrel': {'color': '#7c3aed', 'linestyle': '-', 'linewidth': 2, 'label': 'Kestrel Humidity'},
        'RH V2': {'color': '#be185d', 'linestyle': '-', 'linewidth': 2, 'label': 'V2 Humidity'},
        'RH V3': {'color': '#0891b2', 'linestyle': '-', 'linewidth': 2, 'label': 'V3 Humidity'}
    }
    
    # Helper function to find time index
    def find_time_index(time_str):
        """Find the index corresponding to a specific time"""
        try:
            time_matches = df[df['time'] == time_str]
            if len(time_matches) > 0:
                return time_matches.index[0]
            # If exact match not found, find closest time after
            for i, t in enumerate(df['time']):
                if str(t) >= time_str:
                    return i
            return len(df) - 1
        except Exception as e:
            print(f"Error finding time index for {time_str}: {e}")
            return 0
    
    # Calculate engineering analysis for slat density optimization
    def calculate_slat_optimization_analysis():
        """Analyze 5-slat vs no-screen performance focusing on V3 tracking to TSI reference"""
        try:
            # Define period boundaries (skip thermal equilibration period)
            screens_removed_idx = find_time_index('13:09')
            screen_change_idx = find_time_index('11:28')
            
            print(f"Debug: screen_change_idx = {screen_change_idx}, screens_removed_idx = {screens_removed_idx}")
            print(f"Debug: DataFrame length = {len(df)}")
            
            # Ensure indices are valid
            screen_change_idx = max(0, min(screen_change_idx, len(df)-1))
            screens_removed_idx = max(0, min(screens_removed_idx, len(df)-1))
            
            # Split data into steady-state periods only
            if screens_removed_idx > screen_change_idx:
                period_5slat = df.iloc[screen_change_idx:screens_removed_idx].copy()  # 5 Screen period
                period_5slat_start = df.iloc[screen_change_idx]['time'] if screen_change_idx < len(df) else ''
                period_5slat_end = df.iloc[screens_removed_idx-1]['time'] if screens_removed_idx > 0 else ''
            else:
                period_5slat = pd.DataFrame()
                period_5slat_start = period_5slat_end = ''
            
            if screens_removed_idx < len(df):
                period_noscreen = df.iloc[screens_removed_idx:].copy()  # No Screen period
                period_noscreen_start = df.iloc[screens_removed_idx]['time'] if screens_removed_idx < len(df) else ''
                period_noscreen_end = df.iloc[-1]['time'] if len(df) > 0 else ''
            else:
                period_noscreen = pd.DataFrame()
                period_noscreen_start = period_noscreen_end = ''
            
            analysis = {}
            
            # Reference sensors for tracking analysis
            reference_sensor = 'TA TSI'  # TSI is gospel
            tracking_sensors = ['TA Kestrel', 'TA V3']  # V3 is primary focus, Kestrel for comparison
            
            def calculate_tracking_performance(period_data, period_name):
                """Calculate how well sensors track the TSI reference"""
                if reference_sensor not in period_data.columns:
                    return None
                
                tracking_stats = {}
                reference_data = period_data[reference_sensor].dropna()
                
                if len(reference_data) < 2:
                    return None
                
                for sensor in tracking_sensors:
                    if sensor in period_data.columns:
                        sensor_data = period_data[sensor].dropna()
                        
                        # Align data (only use times where both sensors have readings)
                        aligned_data = period_data[[reference_sensor, sensor]].dropna()
                        if len(aligned_data) < 2:
                            continue
                            
                        ref_aligned = aligned_data[reference_sensor]
                        sensor_aligned = aligned_data[sensor]
                        
                        # Calculate deviations from TSI
                        deviations = (sensor_aligned - ref_aligned).abs()
                        
                        # Tracking performance metrics
                        avg_deviation = deviations.mean()
                        max_deviation = deviations.max()
                        within_1deg = (deviations <= 1.0).sum() / len(deviations) * 100
                        within_2deg = (deviations <= 2.0).sum() / len(deviations) * 100
                        within_3deg = (deviations <= 3.0).sum() / len(deviations) * 100
                        
                        tracking_stats[sensor] = {
                            'avg_deviation': avg_deviation,
                            'max_deviation': max_deviation,
                            'within_1deg_pct': within_1deg,
                            'within_2deg_pct': within_2deg,
                            'within_3deg_pct': within_3deg,
                            'data_points': len(aligned_data),
                            'sensor_mean': sensor_aligned.mean(),
                            'reference_mean': ref_aligned.mean()
                        }
                
                return {
                    'tracking_stats': tracking_stats,
                    'reference_mean': reference_data.mean(),
                    'data_points': len(reference_data)
                }
            
            periods = [
                ('5 Slat Period', period_5slat, period_5slat_start, period_5slat_end),
                ('No Screen Period', period_noscreen, period_noscreen_start, period_noscreen_end)
            ]
            
            for period_name, period_data, start_time, end_time in periods:
                if len(period_data) > 1:
                    tracking_performance = calculate_tracking_performance(period_data, period_name)
                    
                    if tracking_performance:
                        analysis[period_name] = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_minutes': len(period_data) * 5,  # assuming 5-min intervals
                            'reference_temp': tracking_performance['reference_mean'],
                            'tracking_performance': tracking_performance['tracking_stats'],
                            'data_points': tracking_performance['data_points']
                        }
                else:
                    analysis[period_name] = None
            
            # Comparative analysis - focus on V3 performance
            if '5 Slat Period' in analysis and 'No Screen Period' in analysis:
                if analysis['5 Slat Period'] and analysis['No Screen Period']:
                    v3_5slat_stats = analysis['5 Slat Period']['tracking_performance'].get('TA V3')
                    v3_noscreen_stats = analysis['No Screen Period']['tracking_performance'].get('TA V3')
                    
                    if v3_5slat_stats and v3_noscreen_stats:
                        # V3 tracking improvement with 5 slats
                        v3_deviation_improvement = v3_noscreen_stats['avg_deviation'] - v3_5slat_stats['avg_deviation']
                        v3_accuracy_improvement = v3_5slat_stats['within_2deg_pct'] - v3_noscreen_stats['within_2deg_pct']
                        
                        # Engineering assessment
                        if v3_deviation_improvement > 0.5:
                            screen_effectiveness = "EXCELLENT"
                        elif v3_deviation_improvement > 0.1:
                            screen_effectiveness = "GOOD"
                        elif v3_deviation_improvement > -0.2:
                            screen_effectiveness = "ACCEPTABLE"
                        else:
                            screen_effectiveness = "PROBLEMATIC"
                        
                        analysis['V3_Performance'] = {
                            'deviation_improvement': v3_deviation_improvement,
                            'accuracy_improvement': v3_accuracy_improvement,
                            'screen_effectiveness': screen_effectiveness,
                            'v3_5slat_deviation': v3_5slat_stats['avg_deviation'],
                            'v3_noscreen_deviation': v3_noscreen_stats['avg_deviation']
                        }
            
            return analysis
            
        except Exception as e:
            print(f"Error in slat optimization analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    engineering_analysis = calculate_slat_optimization_analysis()
    
    # Figure 1: Ambient Temperatures
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    for sensor, config in ambient_config.items():
        if sensor in df.columns and not df[sensor].isna().all():
            # Calculate moving average
            smoothed = calculate_moving_average(df[sensor])
            
            # Plot raw data as scatter (X markers)
            ax1.scatter(df.index, df[sensor], 
                       marker='x', s=30, 
                       color=config['color'], 
                       alpha=0.6,
                       label=f"{config['label']} (Raw)")
            
            # Plot smoothed trend line
            ax1.plot(df.index, smoothed,
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    label=f"{config['label']} (Trend)")
    
    # Add annotations to ambient temperature plot
    # Vertical line at 11:28 (6Screen to 5Screen change)
    screen_change_idx = find_time_index('11:28')
    ax1.axvline(x=screen_change_idx, color='green', linestyle=':', linewidth=2, alpha=0.8)
    ax1.annotate('5-Slat Period Begins\n(Steady State)', 
                xy=(screen_change_idx, ax1.get_ylim()[1]), 
                xytext=(screen_change_idx + 5, ax1.get_ylim()[1] - 2),
                fontsize=10, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', edgecolor='green', alpha=0.8))
    
    # Annotation at 12:10 (light breeze)
    breeze_idx = find_time_index('12:10')
    ax1.annotate('Light breeze begins\n(ending stagnant period)', 
                xy=(breeze_idx, df.iloc[breeze_idx]['TA V2'] if breeze_idx < len(df) else 80), 
                xytext=(breeze_idx + 8, 85),
                fontsize=10, fontweight='bold', color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', edgecolor='blue', alpha=0.8))
    
    # Vertical line at 13:09 (screens removed)
    screens_removed_idx = find_time_index('13:09')
    ax1.axvline(x=screens_removed_idx, color='purple', linestyle=':', linewidth=2, alpha=0.8)
    ax1.annotate('No Screen Period\n(Control)', 
                xy=(screens_removed_idx, ax1.get_ylim()[1]), 
                xytext=(screens_removed_idx + 5, ax1.get_ylim()[1] - 2),
                fontsize=10, fontweight='bold', color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', edgecolor='purple', alpha=0.8))
    
    # Add engineering analysis display
    if engineering_analysis and any(engineering_analysis.get(k) for k in ['5 Slat Period', 'No Screen Period']):
        # Create analysis text
        analysis_text = "V3 TRACKING PERFORMANCE ANALYSIS:\n(TSI = Reference Truth)\n\n"
        
        for period_name in ['5 Slat Period', 'No Screen Period']:
            if period_name in engineering_analysis and engineering_analysis[period_name]:
                stats = engineering_analysis[period_name]
                analysis_text += f"{period_name} ({stats['start_time']} - {stats['end_time']}):\n"
                analysis_text += f"  • TSI Avg: {stats['reference_temp']:.1f}°F\n"
                
                # V3 tracking performance
                if 'TA V3' in stats['tracking_performance']:
                    v3_stats = stats['tracking_performance']['TA V3']
                    analysis_text += f"  • V3 Dev from TSI: ±{v3_stats['avg_deviation']:.2f}°F\n"
                    analysis_text += f"  • V3 within ±2°F: {v3_stats['within_2deg_pct']:.0f}%\n"
                
                # Kestrel for comparison
                if 'TA Kestrel' in stats['tracking_performance']:
                    k_stats = stats['tracking_performance']['TA Kestrel']
                    analysis_text += f"  • Kestrel Dev: ±{k_stats['avg_deviation']:.2f}°F\n"
                
                analysis_text += f"  • Data Points: {stats['data_points']}\n\n"
        
        # Add comparative analysis
        if 'V3_Performance' in engineering_analysis:
            v3_perf = engineering_analysis['V3_Performance']
            analysis_text += "SCREEN EFFECTIVENESS VERDICT:\n"
            analysis_text += f"  • V3 Improvement: {v3_perf['deviation_improvement']:+.2f}°F\n"
            analysis_text += f"  • Screen Assessment: {v3_perf['screen_effectiveness']}\n"
            
            if v3_perf['deviation_improvement'] > 0.2:
                analysis_text += "  ✅ Screen helps V3 track TSI better\n"
            elif v3_perf['deviation_improvement'] > -0.1:
                analysis_text += "  ⚡ Screen has minimal impact\n"
            else:
                analysis_text += "  ⚠️  Screen makes V3 tracking worse\n"
        
        # Add analysis text box
        ax1.text(0.02, 0.98, analysis_text, 
                transform=ax1.transAxes, 
                fontsize=8, 
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', edgecolor='orange', alpha=0.9),
                family='monospace')
        
        # Print detailed engineering analysis to console
        print("\n" + "="*70)
        print("V3 SENSOR TRACKING PERFORMANCE ANALYSIS")
        print("="*70)
        print("GOAL: Design screen so V3 tracks as close to TSI as possible")
        print("TSI = Reference Truth | Focus: V3 Performance\n")
        
        for period_name in ['5 Slat Period', 'No Screen Period']:
            if period_name in engineering_analysis and engineering_analysis[period_name]:
                stats = engineering_analysis[period_name]
                print(f"{period_name.upper()} ({stats['start_time']} - {stats['end_time']}):")
                print(f"  Duration: {stats['duration_minutes']:.0f} minutes ({stats['data_points']} measurements)")
                print(f"  TSI Reference Average: {stats['reference_temp']:.2f}°F")
                
                # Detailed tracking performance for each sensor
                for sensor_name, tracking_stats in stats['tracking_performance'].items():
                    sensor_short = sensor_name.replace('TA ', '')
                    print(f"\n  {sensor_short} Tracking Performance:")
                    print(f"    Average Deviation from TSI: ±{tracking_stats['avg_deviation']:.3f}°F")
                    print(f"    Maximum Deviation: {tracking_stats['max_deviation']:.2f}°F")
                    print(f"    Time within ±1°F of TSI: {tracking_stats['within_1deg_pct']:.1f}%")
                    print(f"    Time within ±2°F of TSI: {tracking_stats['within_2deg_pct']:.1f}%")
                    print(f"    Time within ±3°F of TSI: {tracking_stats['within_3deg_pct']:.1f}%")
                    print(f"    {sensor_short} Average: {tracking_stats['sensor_mean']:.2f}°F")
                print()
        
        # V3-focused verdict
        if 'V3_Performance' in engineering_analysis:
            v3_perf = engineering_analysis['V3_Performance']
            print("V3 SCREEN DESIGN EFFECTIVENESS:")
            print(f"V3 Deviation with 5 Slats: ±{v3_perf['v3_5slat_deviation']:.3f}°F from TSI")
            print(f"V3 Deviation with No Screen: ±{v3_perf['v3_noscreen_deviation']:.3f}°F from TSI")
            print(f"V3 Tracking Improvement: {v3_perf['deviation_improvement']:+.3f}°F")
            print(f"  (Positive = V3 tracks TSI better with 5 slats)")
            print(f"Engineering Assessment: {v3_perf['screen_effectiveness']}")
            
            print("\nDESIGN RECOMMENDATION:")
            if v3_perf['deviation_improvement'] > 0.3:
                print("✅ EXCELLENT: 5-slat screen significantly improves V3 tracking accuracy")
            elif v3_perf['deviation_improvement'] > 0.1:
                print("✅ GOOD: 5-slat screen moderately improves V3 tracking")
            elif v3_perf['deviation_improvement'] > -0.1:
                print("⚡ ACCEPTABLE: 5-slat screen has minimal impact on V3 tracking")
            else:
                print("⚠️  PROBLEMATIC: 5-slat screen degrades V3 tracking - consider design changes")
                
        print("="*70)
    
    # Customize ambient temperature plot
    ax1.set_ylabel('Temperature (°F)', fontsize=12, fontweight='bold')
    ax1.set_title('BEACON Screen Testing - Ambient Temperatures (8/15/2025)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set x-axis labels
    tick_indices = range(0, len(df), 5)
    tick_labels = [df.loc[i, 'time'] if i < len(df) else '' for i in tick_indices]
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    figures.append(fig1)
    
    
    # Figure 2: Globe Temperatures
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    for sensor, config in globe_config.items():
        if sensor in df.columns and not df[sensor].isna().all():
            # Calculate moving average
            smoothed = calculate_moving_average(df[sensor])
            
            # Plot raw data as scatter (X markers)
            ax2.scatter(df.index, df[sensor], 
                       marker='x', s=30, 
                       color=config['color'], 
                       alpha=0.6,
                       label=f"{config['label']} (Raw)")
            
            # Plot smoothed trend line
            ax2.plot(df.index, smoothed,
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    label=f"{config['label']} (Trend)")
    
    # Customize globe temperature plot
    ax2.set_ylabel('Temperature (°F)', fontsize=12, fontweight='bold')
    ax2.set_title('BEACON Screen Testing - Globe (Radiant) Temperatures (8/15/2025)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set x-axis labels
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    figures.append(fig2)
    
    
    # Figure 3: Humidity
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    for sensor, config in humidity_config.items():
        if sensor in df.columns and not df[sensor].isna().all():
            # Calculate moving average
            smoothed = calculate_moving_average(df[sensor])
            
            # Plot raw data as scatter (X markers)
            ax3.scatter(df.index, df[sensor], 
                       marker='x', s=30, 
                       color=config['color'], 
                       alpha=0.6,
                       label=f"{config['label']} (Raw)")
            
            # Plot smoothed trend line
            ax3.plot(df.index, smoothed,
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    label=f"{config['label']} (Trend)")
    
    # Customize humidity plot
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Humidity (%)', fontsize=12, fontweight='bold')
    ax3.set_title('BEACON Screen Testing - Relative Humidity (8/15/2025)', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set x-axis labels
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    figures.append(fig3)
    
    return figures

def create_interactive_plot(df):
    """Create an interactive plot using plotly (optional enhancement)"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Ambient Temperatures', 'Globe (Radiant) Temperatures', 'Relative Humidity'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Ambient temperatures
        ambient_sensors = ['TA TSI', 'TA Kestrel', 'TA V2', 'TA V3']
        colors_ambient = ['red', 'blue', 'black', 'orange']
        
        for i, sensor in enumerate(ambient_sensors):
            if sensor in df.columns and not df[sensor].isna().all():
                # Raw data
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[sensor],
                    mode='markers',
                    marker=dict(symbol='x', size=8, color=colors_ambient[i]),
                    name=f"{sensor.replace('TA ', '')} (Raw)",
                    opacity=0.7
                ), row=1, col=1)
                
                # Smoothed trend
                smoothed = calculate_moving_average(df[sensor])
                fig.add_trace(go.Scatter(
                    x=df['time'], y=smoothed,
                    mode='lines',
                    line=dict(color=colors_ambient[i], width=2),
                    name=f"{sensor.replace('TA ', '')} (Trend)"
                ), row=1, col=1)
        
        # Globe temperatures
        globe_sensors = ['TG TSI', 'TG Kestrel']
        colors_globe = ['red', 'blue']
        
        for i, sensor in enumerate(globe_sensors):
            if sensor in df.columns and not df[sensor].isna().all():
                # Raw data
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[sensor],
                    mode='markers',
                    marker=dict(symbol='x', size=8, color=colors_globe[i]),
                    name=f"{sensor.replace('TG ', '')} Globe (Raw)",
                    opacity=0.7
                ), row=2, col=1)
                
                # Smoothed trend
                smoothed = calculate_moving_average(df[sensor])
                fig.add_trace(go.Scatter(
                    x=df['time'], y=smoothed,
                    mode='lines',
                    line=dict(color=colors_globe[i], width=2, dash='dash'),
                    name=f"{sensor.replace('TG ', '')} Globe (Trend)"
                ), row=2, col=1)
        
        # Humidity
        humidity_sensors = ['RH TSI', 'RH Kestrel', 'RH V2', 'RH V3']
        colors_humidity = ['green', 'purple', 'pink', 'cyan']
        
        for i, sensor in enumerate(humidity_sensors):
            if sensor in df.columns and not df[sensor].isna().all():
                # Raw data
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[sensor],
                    mode='markers',
                    marker=dict(symbol='x', size=8, color=colors_humidity[i]),
                    name=f"{sensor.replace('RH ', '')} RH (Raw)",
                    opacity=0.7
                ), row=3, col=1)
                
                # Smoothed trend
                smoothed = calculate_moving_average(df[sensor])
                fig.add_trace(go.Scatter(
                    x=df['time'], y=smoothed,
                    mode='lines',
                    line=dict(color=colors_humidity[i], width=2),
                    name=f"{sensor.replace('RH ', '')} RH (Trend)"
                ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title='BEACON Screen Testing - 8/15/2025 Temperature & Humidity Data',
            height=1000,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°F)", row=2, col=1)
        fig.update_yaxes(title_text="Relative Humidity (%)", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig
        
    except ImportError:
        print("Plotly not available. Using matplotlib only.")
        return None

# Main execution
if __name__ == "__main__":
    # Update this path to your CSV file
    csv_file_path = "8_15_2025_TEMPDATA.csv"
    
    try:
        # Process the data
        df = load_and_process_data(csv_file_path)
        
        print(f"Data loaded successfully!")
        print(f"Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        print(f"Number of measurements: {len(df)}")
        
        # Create matplotlib visualization
        figures = create_temperature_visualization(df)
        plt.show()
        
        # Optional: Create interactive plotly version
        plotly_fig = create_interactive_plot(df)
        if plotly_fig:
            plotly_fig.show()
            
        # Print engineering summary statistics
        print("\nSLAT DENSITY ENGINEERING SUMMARY:")
        print("="*50)
        
        if engineering_analysis and 'Comparative' in engineering_analysis:
            comp = engineering_analysis['Comparative']
            print(f"Heat Trapping Effect: {comp['heat_trapping_effect']:+.2f}°F")
            print(f"Measurement Quality Impact: {comp['measurement_quality_change']:+.2f}°F")
            print(f"Engineering Assessment: {comp['slat_effectiveness']}")
        else:
            print("Engineering analysis not available - check data quality")
            
        print("\nIndividual Sensor Averages (Steady-State Periods Only):")
        ambient_sensors = ['TA TSI', 'TA Kestrel', 'TA V2', 'TA V3']
        for sensor in ambient_sensors:
            if sensor in df.columns and not df[sensor].isna().all():
                # Calculate averages for steady-state periods only
                screen_change_idx = find_time_index('11:28')
                steady_state_data = df.iloc[screen_change_idx:][sensor].dropna()
                if len(steady_state_data) > 0:
                    print(f"  {sensor}: {steady_state_data.mean():.1f}°F (±{steady_state_data.std():.1f}°F)")
                
        print("="*50)
                
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
        print("Please update the csv_file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"Error processing data: {e}")

# Additional utility functions

def save_high_quality_plot(df, base_filename="beacon_temperature_analysis_8_15_2025", dpi=300):
    """Save high-quality versions of all plots"""
    figures = create_temperature_visualization(df)
    plot_names = ["_ambient_temps", "_globe_temps", "_humidity"]
    
    for i, fig in enumerate(figures):
        filename = f"{base_filename}{plot_names[i]}.png"
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"High-quality plot saved as {filename}")
        plt.close(fig)  # Close figure to free memory

def export_processed_data(df, filename="beacon_processed_data_8_15_2025.csv"):
    """Export the processed data with moving averages"""
    # Calculate moving averages for all sensors
    all_sensors = ['TA TSI', 'TA Kestrel', 'TA V2', 'TA V3', 'TG TSI', 'TG Kestrel', 
                   'RH TSI', 'RH Kestrel', 'RH V2', 'RH V3']
    export_df = df[['time'] + [s for s in all_sensors if s in df.columns]].copy()
    
    for sensor in all_sensors:
        if sensor in df.columns:
            export_df[f"{sensor}_5pt_avg"] = calculate_moving_average(df[sensor])
    
    export_df.to_csv(filename, index=False)
    print(f"Processed data exported to {filename}")

# Example usage for additional features:
# save_high_quality_plot(df, "my_temperature_plots")  # Will create 3 separate PNG files
# export_processed_data(df, "processed_temperature_data.csv")