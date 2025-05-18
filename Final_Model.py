import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from collections import Counter

# 1) Load the saved artifacts and model
model = tf.keras.models.load_model('battery_classifier.keras')
scaler_battery = joblib.load('scaler_battery.pkl')
model_selector = joblib.load('model_selector.pkl')

# 2) Recompute (or hard-code) your training min/max freq for normalization:
training_data = pd.read_excel(r'Training_Data/DATA_MODEL.xlsx')
min_freq = training_data['Frequency'].min()
max_freq = training_data['Frequency'].max()

# 3) Constants
EPS = 1e-8
num_datapoints_per_battery = 75
label_map = {0: "New", 1: "Used", 2: "Degraded"}

# 4) Para same features ulit sa training data


def predict_from_saved(file_path):
    df = pd.read_excel(file_path)

    # 4a) basic features + engineered ones
    X = pd.DataFrame({
        'Real':      df['Real'],
        'Imaginary': df['Imaginary'],
    })
    X['Impedance_Magnitude'] = np.sqrt(X['Real']**2 + X['Imaginary']**2)
    X['Phase_Angle'] = np.arctan2(df['Imaginary'], df['Real'])
    X['Admittance_Real'] = df['Real'] / \
        (df['Real']**2 + df['Imaginary']**2 + EPS)
    X['Admittance_Imag'] = -df['Imaginary'] / \
        (df['Real']**2 + df['Imaginary']**2 + EPS)
    X['Log_Frequency'] = np.log10(df['Frequency'] + EPS)
    X['Log_Impedance'] = np.log10(X['Impedance_Magnitude'] + EPS)
    X['Normalized_Frequency'] = (
        df['Frequency'] - min_freq) / (max_freq - min_freq + EPS)

    # 4b) battery-level stats
    feats = {
        'Mean_Real':      X['Real'].mean(),
        'Std_Real':       X['Real'].std(),
        'Mean_Imaginary': X['Imaginary'].mean(),
        'Std_Imaginary':  X['Imaginary'].std(),
        'Mean_Magnitude': X['Impedance_Magnitude'].mean(),
        'Max_Real':       X['Real'].max(),
        'Min_Real':       X['Real'].min(),
        'Max_Imaginary':  X['Imaginary'].max(),
        'Min_Imaginary':  X['Imaginary'].min(),
        'Slope_Real_Frequency': np.polyfit(np.log10(df['Frequency']+EPS), df['Real'], 1)[0],
        'Slope_Imag_Frequency': np.polyfit(np.log10(df['Frequency']+EPS), df['Imaginary'], 1)[0],
        'Low_Freq_Real_Mean':  X.loc[df['Frequency'] < 10,      'Real'].mean(),
        'High_Freq_Real_Mean': X.loc[df['Frequency'] > 1000,    'Real'].mean(),
        'Low_Freq_Imag_Mean':  X.loc[df['Frequency'] < 10,      'Imaginary'].mean(),
        'High_Freq_Imag_Mean': X.loc[df['Frequency'] > 1000,    'Imaginary'].mean(),
    }
    # ratios
    feats['Real_Low_High_Ratio'] = (
        feats['Low_Freq_Real_Mean'] / (feats['High_Freq_Real_Mean'] + EPS))
    feats['Imag_Low_High_Ratio'] = (
        feats['Low_Freq_Imag_Mean'] / (feats['High_Freq_Imag_Mean'] + EPS))

    # 4c) scale and select
    batch = pd.DataFrame([feats])
    scaled = scaler_battery.transform(batch)
    sel = model_selector.transform(scaled)

    # 4d) predict
    probs = model.predict(sel, verbose=0)[0]
    idx = np.argmax(probs)
    return label_map[idx], probs[idx]


# 5) Run predictions and collect detailed results
evaluation_dir = 'Evaluation_Data'
results = {
    'New':      {'files': [], 'correct': 0},
    'Used':     {'files': [], 'correct': 0},
    'Degraded': {'files': [], 'correct': 0}
}

for label_folder in ['new', 'used', 'degraded']:
    expected = label_folder.capitalize()
    folder = os.path.join(evaluation_dir, label_folder)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.xlsx'):
            continue
        path = os.path.join(folder, fname)
        pred, conf = predict_from_saved(path)
        is_correct = (pred == expected)

        results[expected]['files'].append({
            'name':    fname,
            'pred':    pred,
            'conf':    conf,
            'correct': is_correct
        })
        if is_correct:
            results[expected]['correct'] += 1

# 6) Print summary breakdown
total_files = sum(len(data['files']) for data in results.values())
total_correct = sum(data['correct'] for data in results.values())

print("\n=== SUMMARY ===\n")
for true_label, data in results.items():
    files = data['files']
    n = len(files)
    c = data['correct']
    pred_counts = Counter(f['pred'] for f in files)

    print(f"{true_label}: processed {n} files, {c}/{n} correct ({c/n:.2%})")
    for f in files:
        mark = '✓' if f['correct'] else '✗'
        print(
            f"  {mark} {f['name']:20s} → expected {true_label:8s} : predicted {f['pred']:8s} (conf {f['conf']:.2f})")
    print("  Prediction breakdown:")
    for lbl, cnt in pred_counts.items():
        print(f"    {lbl:8s}: {cnt}/{n} ({cnt/n:.2%})")
    print()

print(
    f"TOTAL: {total_correct}/{total_files} correct ({total_correct/total_files:.2%})")
overall_counts = Counter(
    pred for data in results.values() for pred in (f['pred'] for f in data['files'])
)
print("Overall prediction breakdown:")
for lbl, cnt in overall_counts.items():
    print(f"  {lbl:8s}: {cnt}/{total_files} ({cnt/total_files:.2%})")
