import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 1) Load saved artifacts and model
model = tf.keras.models.load_model('battery_classifier.keras')
scaler_battery = joblib.load('scaler_battery.pkl')
model_selector = joblib.load('model_selector.pkl')

# 2) Recompute your training min/max freq for normalization:
training_data = pd.read_excel(r'Training_Data/DATA_MODEL.xlsx')
min_freq = training_data['Frequency'].min()
max_freq = training_data['Frequency'].max()

# 3) Constants
EPS = 1e-8  # number close to zero para ma avoid mag divide by zero if ever
num_datapoints_per_battery = 75
label_map = {0: "New", 1: "Used", 2: "Degraded"}


def extract_features(df):
    """Extract features from an Excel file with battery data"""
    # Basic features
    X = pd.DataFrame({
        'Real': df['Real'],
        'Imaginary': df['Imaginary'],
        'Frequency': df['Frequency']
    })

    # Calculated Features
    X['Impedance_Magnitude'] = np.sqrt(X['Real']**2 + X['Imaginary']**2)
    X['Phase_Angle'] = np.arctan2(X['Imaginary'], X['Real'])
    X['Admittance_Real'] = X['Real'] / (X['Real']**2 + X['Imaginary']**2 + EPS)
    X['Admittance_Imag'] = -X['Imaginary'] / \
        (X['Real']**2 + X['Imaginary']**2 + EPS)
    X['Log_Frequency'] = np.log10(X['Frequency'] + EPS)
    X['Log_Impedance'] = np.log10(X['Impedance_Magnitude'] + EPS)
    X['Normalized_Frequency'] = (
        X['Frequency'] - min_freq) / (max_freq - min_freq + EPS)

    # Battery-level statistical features
    feats = {
        'Mean_Real': X['Real'].mean(),
        'Std_Real': X['Real'].std(),
        'Mean_Imaginary': X['Imaginary'].mean(),
        'Std_Imaginary': X['Imaginary'].std(),
        'Mean_Magnitude': X['Impedance_Magnitude'].mean(),
        'Max_Real': X['Real'].max(),
        'Min_Real': X['Real'].min(),
        'Max_Imaginary': X['Imaginary'].max(),
        'Min_Imaginary': X['Imaginary'].min(),
        'Slope_Real_Frequency': np.polyfit(np.log10(X['Frequency']+EPS), X['Real'], 1)[0],
        'Slope_Imag_Frequency': np.polyfit(np.log10(X['Frequency']+EPS), X['Imaginary'], 1)[0],
        'Low_Freq_Real_Mean': X.loc[X['Frequency'] < 10, 'Real'].mean(),
        'High_Freq_Real_Mean': X.loc[X['Frequency'] > 1000, 'Real'].mean(),
        'Low_Freq_Imag_Mean': X.loc[X['Frequency'] < 10, 'Imaginary'].mean(),
        'High_Freq_Imag_Mean': X.loc[X['Frequency'] > 1000, 'Imaginary'].mean(),
    }

    # Ratio features
    feats['Real_Low_High_Ratio'] = feats['Low_Freq_Real_Mean'] / \
        (feats['High_Freq_Real_Mean'] + EPS)
    feats['Imag_Low_High_Ratio'] = feats['Low_Freq_Imag_Mean'] / \
        (feats['High_Freq_Imag_Mean'] + EPS)

    return X, feats


def predict_from_saved(file_path):
    """Predict battery state from Excel file and return features and visualization data"""
    df = pd.read_excel(file_path)

    # Extract features
    X, feats = extract_features(df)

    # Scale and select features
    batch = pd.DataFrame([feats])
    scaled = scaler_battery.transform(batch)
    sel = model_selector.transform(scaled)

    # Get feature vectors
    feature_vector_raw = pd.DataFrame([feats])
    feature_vector_scaled = pd.DataFrame(scaled, columns=batch.columns)
    feature_vector_selected = pd.DataFrame(sel)

    # Predict
    probs = model.predict(sel, verbose=0)[0]
    idx = np.argmax(probs)

    # Get all probabilities for all classes
    class_probs = {label_map[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "prediction": label_map[idx],
        "confidence": float(probs[idx]),
        "class_probabilities": class_probs,
        "feature_vectors": {
            "raw": feature_vector_raw.to_dict('records')[0],
            "scaled": feature_vector_scaled.to_dict('records')[0],
            "selected": feature_vector_selected.to_dict('records')[0]
        },
        "visualization_data": {
            "frequency": df['Frequency'].tolist(),
            "real": df['Real'].tolist(),
            "imaginary": df['Imaginary'].tolist(),
            "impedance_magnitude": X['Impedance_Magnitude'].tolist(),
            "phase_angle": X['Phase_Angle'].tolist()
        }
    }


def plot_battery_data(data, title="Battery EIS Analysis"):
    """Create Bode and Nyquist plots from battery data"""
    freq = data["visualization_data"]["frequency"]
    real = data["visualization_data"]["real"]
    imag = data["visualization_data"]["imaginary"]
    mag = data["visualization_data"]["impedance_magnitude"]
    phase = data["visualization_data"]["phase_angle"]

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Nyquist plot (-Imag vs Real)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(-np.array(imag), real, 'o-', markersize=5)
    ax1.set_xlabel('Z\' (Real) [Ω]')
    ax1.set_ylabel('-Z\" (Imaginary) [Ω]')
    ax1.set_title('Nyquist Plot')
    ax1.grid(True)

    # Bode plot - Magnitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(freq, mag, 'o-', markersize=5)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('|Z| [Ω]')
    ax2.set_title('Bode Plot - Impedance Magnitude')
    ax2.grid(True, which="both")

    # Bode plot - Phase
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogx(freq, np.degrees(phase), 'o-', markersize=5)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Phase [°]')
    ax3.set_title('Bode Plot - Phase Angle')
    ax3.grid(True, which="both")

    # Add prediction result to the overall title
    prediction = data["prediction"]
    confidence = data["confidence"]
    main_title = f"{title}\nPredicted: {prediction} (Confidence: {confidence:.2f})"
    fig.suptitle(main_title, fontsize=16)

    plt.tight_layout()
    return fig


def display_bode_vectors(data, num_points=75):
    """
    Display the vectors used in creating the Bode plot

    Parameters:
    -----------
    data : dict
        Dictionary containing visualization data from predict_from_saved function
    num_points : int, optional
        Number of data points to display (default: 75)
    """
    # Extract the relevant vectors from the data
    freq = data["visualization_data"]["frequency"]
    mag = data["visualization_data"]["impedance_magnitude"]
    phase = data["visualization_data"]["phase_angle"]
    real = data["visualization_data"]["real"]
    imag = data["visualization_data"]["imaginary"]

    # Convert phase from radians to degrees for display
    phase_degrees = np.degrees(phase)

    # Create a DataFrame for better display
    df = pd.DataFrame({
        'Frequency (Hz)': freq,
        'Impedance |Z| (Ω)': mag,
        'Phase Angle (°)': phase_degrees,
        'Real Z\' (Ω)': real,
        'Imaginary Z\" (Ω)': imag
    })

    # Sort by frequency to ensure data is in order
    df = df.sort_values(by='Frequency (Hz)')

    # Select a subset of points to display
    if len(df) > num_points:
        # Take points spread evenly across the frequency range
        indices = np.linspace(0, len(df)-1, num_points, dtype=int)
        df_display = df.iloc[indices]
    else:
        df_display = df

    # Display the data
    print(
        f"\n=== BODE PLOT VECTORS (showing {len(df_display)} of {len(df)} points) ===")
    print("\nData used for Bode and Nyquist plots:")
    print(df_display.to_string(index=False, float_format=lambda x: f"{x:.4e}" if abs(
        x) > 1000 or abs(x) < 0.01 else f"{x:.4f}"))

    # Display some statistics
    print("\nVector Statistics:")
    print(f"Frequency Range: {min(freq):.2e} Hz to {max(freq):.2e} Hz")
    print(f"Impedance Magnitude Range: {min(mag):.4f} Ω to {max(mag):.4f} Ω")
    print(
        f"Phase Angle Range: {min(phase_degrees):.2f}° to {max(phase_degrees):.2f}°")

    return df


def run_batch_predictions():
    """Run predictions on all evaluation data and collect results"""
    evaluation_dir = 'Evaluation_Data'
    results = {
        'New': {'files': [], 'correct': 0, 'before': 0, 'after': 0},
        'Used': {'files': [], 'correct': 0, 'before': 0, 'after': 0},
        'Degraded': {'files': [], 'correct': 0, 'before': 0, 'after': 0}
    }

    all_data = []

    for label_folder in ['new', 'used', 'degraded']:
        expected = label_folder.capitalize()
        folder = os.path.join(evaluation_dir, label_folder)

        # Count files before processing
        xlsx_files = [f for f in os.listdir(
            folder) if f.lower().endswith('.xlsx')]
        results[expected]['before'] = len(xlsx_files)

        for fname in sorted(xlsx_files):
            path = os.path.join(folder, fname)
            result_data = predict_from_saved(path)
            pred = result_data["prediction"]
            conf = result_data["confidence"]
            is_correct = (pred == expected)

            # Store full result data
            file_result = {
                'filename': fname,
                'path': path,
                'expected': expected,
                'predicted': pred,
                'confidence': conf,
                'correct': is_correct,
                'data': result_data
            }

            results[expected]['files'].append(file_result)
            all_data.append(file_result)

            if is_correct:
                results[expected]['correct'] += 1

        # Count files after processing
        results[expected]['after'] = len(results[expected]['files'])

    return results, all_data


def display_battery_analysis(file_path=None, result_data=None):
    """Display comprehensive analysis for a single battery file"""
    if file_path is not None and result_data is None:
        result_data = predict_from_saved(file_path)

    # Extract the filename from the path if available
    title = os.path.basename(file_path) if file_path else "Battery Analysis"

    # Create plots
    fig = plot_battery_data(result_data, title)

    vectors_df = display_bode_vectors(result_data)

    # Display feature importance
    print(f"\n=== PREDICTION RESULTS: {title} ===")
    print(f"Predicted class: {result_data['prediction']}")
    print(f"Confidence: {result_data['confidence']:.4f}")
    print("\nClass probabilities:")
    for label, prob in result_data['class_probabilities'].items():
        print(f"  {label}: {prob:.4f}")

    print("\n=== FEATURE VECTORS ===")
    print("\nTop 10 Raw Features:")
    for i, (k, v) in enumerate(sorted(result_data['feature_vectors']['raw'].items(),
                                      key=lambda x: abs(x[1]), reverse=True)[:10]):
        print(f"  {k}: {v:.4f}")

    print("\nTop Selected Features (after feature selection):")
    for i, (k, v) in enumerate(sorted(result_data['feature_vectors']['selected'].items(),
                                      key=lambda x: abs(x[1]), reverse=True)[:10]):
        print(f"  Feature_{k}: {v:.4f}")

    return fig


def print_summary_results(results):
    """Print summary of all battery predictions"""
    total_files = sum(len(data['files']) for data in results.values())
    total_correct = sum(data['correct'] for data in results.values())

    print("\n=== SUMMARY RESULTS ===\n")
    print("Before and After Processing Counts:")
    for true_label, data in results.items():
        print(
            f"  {true_label}: Before={data['before']} files, After={data['after']} files")
    print()

    for true_label, data in results.items():
        files = data['files']
        n = len(files)
        c = data['correct']
        pred_counts = Counter(f['predicted'] for f in files)

        print(f"{true_label}: processed {n} files, {c}/{n} correct ({c/n:.2%})")
        for f in files:
            mark = '✓' if f['correct'] else '✗'
            print(
                f"  {mark} {f['filename']:20s} → expected {true_label:8s} : predicted {f['predicted']:8s} (conf {f['confidence']:.2f})")

        print("  Prediction breakdown:")
        for lbl, cnt in pred_counts.items():
            print(f"    {lbl:8s}: {cnt}/{n} ({cnt/n:.2%})")
        print()

    print(
        f"TOTAL: {total_correct}/{total_files} correct ({total_correct/total_files:.2%})")
    overall_counts = Counter(pred for data in results.values()
                             for pred in (f['predicted'] for f in data['files']))
    print("Overall prediction breakdown:")
    for lbl, cnt in overall_counts.items():
        print(f"  {lbl:8s}: {cnt}/{total_files} ({cnt/total_files:.2%})")


def create_comparison_plots(results):
    """Create comparison plots for New, Used, and Degraded batteries"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['New', 'Used', 'Degraded']

    for i, label in enumerate(labels):
        ax = axes[i]

        # Get one example from each category
        if results[label]['files']:
            example = results[label]['files'][0]
            data = example['data']

            # Plot Nyquist
            real = data["visualization_data"]["real"]
            imag = data["visualization_data"]["imaginary"]
            ax.plot(-np.array(imag), real, 'o-', markersize=5)
            ax.set_xlabel('Z\' (Real) [Ω]')
            ax.set_ylabel('-Z\" (Imaginary) [Ω]')
            ax.set_title(f'Nyquist Plot - {label} Battery')
            ax.grid(True)

    plt.tight_layout()
    return fig


def main():
    print("=== BATTERY CLASSIFICATION SYSTEM ===\n")
    print("1. Running batch predictions on all evaluation data...")
    results, all_data = run_batch_predictions()

    print("2. Generating summary results...")
    print_summary_results(results)

    print("\n3. Displaying analysis for example batteries...\n")

    # Analysis for one battery from each category
    for category in ['New', 'Used', 'Degraded']:
        if results[category]['files']:
            # 0 here means first battery on each category
            example = results[category]['files'][0]
            print(f"\n\n=== EXAMPLE ANALYSIS: {category} BATTERY ===")
            fig = display_battery_analysis(result_data=example['data'])

            # Display a subset of the Bode vectors for this example
            print(f"\nBode vectors for example {category} battery:")
            display_bode_vectors(example['data'], num_points=75)

            plt.figure(fig.number)
            plt.savefig(f"example_{category.lower()}_battery.png")

    print("\n4. Creating comparison plots...")
    comparison_fig = create_comparison_plots(results)
    plt.figure(comparison_fig.number)
    plt.savefig("battery_comparison.png")

    print("\nAnalysis complete! Visualizations saved as PNG files.")
    print("\nTo analyze a specific battery file, use:")
    print("display_battery_analysis('path/to/your/battery_file.xlsx')")


if __name__ == "__main__":
    main()
