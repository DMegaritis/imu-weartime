"""
Final 1D CNN model training and evaluation using LOSO cross-validation.
Optimized for 24-core/512GB RAM Linux VM with NPZ files.

Loads each fold's data from NPZ into RAM for fast training.
Saves training histories and generates ROC, PR, calibration, loss and accuracy curves.
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import json
import gc
import time
from datetime import timedelta
import sys

# Import utilities
from features_novel_AI_model.model_evaluation.utils.windows_to_weartime import overlapping_windows_to_sample_labels
from features_novel_AI_model.model_evaluation.utils.load_ref import get_subject_sensor_data
from analysis_weartime.utils.performance_metrics import wtd_performance_metrics, wtd_performance_metrics_waking_hours

# SET THREADING ENVIRONMENT VARIABLES BEFORE IMPORTING TENSORFLOW
os.environ['TF_NUM_INTRAOP_THREADS'] = '24'
os.environ['TF_NUM_INTEROP_THREADS'] = '24'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Configure TensorFlow for 24-core CPU optimization
import tensorflow as tf

# Disable GPU completely
tf.config.set_visible_devices([], 'GPU')

# Optimize for 24 CPU cores
tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)

print(f"[DEBUG] TensorFlow configured for 24-core CPU")
print(f"[DEBUG] TensorFlow version: {tf.__version__}")
sys.stdout.flush()

# Subject groups
IDs_a = ['001', '002', '003', '004', '005', '006', '007', '008', '009', "010", "011", "012", "013", "014"]
IDs_b = ['020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031']

# Loading the metadata
with open('ref_metadata.json', 'r') as file:
    ref_metadata = json.load(file)

# Load data paths - VM paths
part_a_path = "/mnt/SFData/SUSTAIN_data/cnn_windowed_dataset_lowback_scaled_perwindow.npz"
part_b_path = "/mnt/SFData/SUSTAIN_data/cnn_windowed_dataset_part_b_scaled_perwindow.npz"

# Verify paths exist
if not Path(part_a_path).exists():
    raise FileNotFoundError(f"Part A file not found: {part_a_path}")
if not Path(part_b_path).exists():
    raise FileNotFoundError(f"Part B file not found: {part_b_path}")

print("Loading metadata for LOSO setup...")

# LOAD ALL DATA INTO RAM ONCE (instead of per-fold)
print("\n" + "=" * 60)
print("LOADING ALL DATA INTO RAM (ONE-TIME LOAD)")
print("=" * 60)
data_load_start = time.time()
sys.stdout.flush()

print("Loading Part A data...")
sys.stdout.flush()
part_a_data = np.load(part_a_path)
X_a = part_a_data['X'].astype(np.float32)
groups_a = part_a_data['groups']
labels_a = part_a_data['y']
print(f"  Part A loaded: {X_a.shape}, {X_a.nbytes / 1024**3:.1f} GB")
sys.stdout.flush()

print("Loading Part B data...")
sys.stdout.flush()
part_b_data = np.load(part_b_path)
X_b = part_b_data['X'].astype(np.float32)
groups_b = part_b_data['groups']
labels_b = part_b_data['y']
print(f"  Part B loaded: {X_b.shape}, {X_b.nbytes / 1024**3:.1f} GB")
sys.stdout.flush()

# Combine all data
X_combined = np.concatenate([X_a, X_b], axis=0)
groups_combined = np.concatenate([groups_a, groups_b])
labels_combined = np.concatenate([labels_a, labels_b])

data_load_elapsed = time.time() - data_load_start
print(f"\n✓ ALL DATA LOADED in {timedelta(seconds=int(data_load_elapsed))}")
print(f"  Total shape: {X_combined.shape}")
print(f"  Total memory: {X_combined.nbytes / 1024**3:.1f} GB")
print(f"  Number of subjects: {len(np.unique(groups_combined))}")
print("=" * 60 + "\n")
sys.stdout.flush()

# Get input shape
input_shape = X_combined.shape[1:]

print(f"Dataset info: {len(groups_combined):,} total samples, input shape: {input_shape}")
print(f"Number of subjects: {len(np.unique(groups_combined))}")


def create_model(input_shape, num_conv_layers, filters, kernel_size, pool_size,
                 dropout_rate, dense_units, learning_rate, lstm_units=0):
    """Create CNN or CNN+LSTM model."""
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN feature extraction
    for i in range(num_conv_layers):
        model.add(layers.Conv1D(filters[i], kernel_size, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size))
        model.add(layers.Dropout(dropout_rate))

    # Adding LSTM
    if lstm_units > 0:
        model.add(layers.LSTM(lstm_units, return_sequences=False))
        model.add(layers.Dropout(dropout_rate))
    else:
        model.add(layers.Flatten())

    # Dense head
    model.add(layers.Dense(dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Slightly safer for LSTM
    clipnorm = 1.0 if lstm_units > 0 else None

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Best parameters
best_params = {
    'num_conv_layers': 3,
    'filters': [32, 64, 128],
    'kernel_size': 9,
    'pool_size': 2,
    'dropout_rate': 0.3,
    'dense_units': 64,
    'learning_rate': 0.001,
    'batch_size': 1024,
    'epochs': 100,
    'lstm_units': 64
}

# LOSO cross-validation
gkf = GroupKFold(n_splits=len(np.unique(groups_combined)))

# Storage for results
fold_results = []
all_y_test = []
all_y_pred = []
all_y_prob = []      # for ROC, PR, calibration curves
all_histories = []   # for loss/accuracy curves

print("\n" + "=" * 60)
print("STARTING LOSO CROSS-VALIDATION (26 folds)")
print("=" * 60 + "\n")

total_start = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels_combined, labels_combined, groups_combined)):
    fold_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx + 1}/26")
    print(f"{'=' * 60}")

    # Fix: convert integer ID (e.g. 1, 25) to zero-padded string (e.g. '001', '025')
    test_subject = str(groups_combined[test_idx][0]).zfill(3)
    print(f"Test subject: {test_subject}")
    print(f"Train samples: {len(train_idx):,}, Test samples: {len(test_idx):,}")
    sys.stdout.flush()

    # Check if already completed
    filepath_a = f"/mnt/SFData/SUSTAIN_outputs/WTD/outputs_a/wearable/WTD/WTD_cnn_lstm/{test_subject}__LowerBack.json"
    filepath_b = f"/mnt/SFData/SUSTAIN_outputs/WTD/outputs_b/wearable/WTD/WTD_cnn_lstm/{test_subject}__LowerBack.json"

    if Path(filepath_a).exists() or Path(filepath_b).exists():
        print(f"Fold {fold_idx + 1} ({test_subject}) already completed - loading saved results")

        # Load the saved JSON
        filepath = filepath_a if Path(filepath_a).exists() else filepath_b
        with open(filepath, 'r') as f:
            saved_results = json.load(f)

        # Extract saved predictions if they exist
        if 'predictions' in saved_results:
            # Split indices for part A and B
            n_samples_a = len(groups_a)
            test_idx_a = test_idx[test_idx < n_samples_a]
            test_idx_b = test_idx[test_idx >= n_samples_a] - n_samples_a

            # Get test data for this subject
            test_indices = np.concatenate([test_idx_a, test_idx_b + len(X_a)])
            y_test = labels_combined[test_indices]
            y_pred = np.array(saved_results['predictions']['y_pred'])
            y_prob = np.array(saved_results['predictions']['y_prob'])

            # Add to aggregated lists
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

            # Load training history if it exists
            if 'training_history' in saved_results:
                all_histories.append(saved_results['training_history'])

            print(f"  Loaded {len(y_test)} test samples with predictions and probabilities")
        else:
            print(f"  Warning: No predictions found in saved file - skipping plot data")

        continue

    # Split indices for part A and B
    n_samples_a = len(groups_a)
    train_idx_a = train_idx[train_idx < n_samples_a]
    train_idx_b = train_idx[train_idx >= n_samples_a] - n_samples_a
    test_idx_a = test_idx[test_idx < n_samples_a]
    test_idx_b = test_idx[test_idx >= n_samples_a] - n_samples_a

    # Train/val split
    np.random.seed(42)
    train_idx_a_shuffled = np.random.permutation(train_idx_a)
    train_idx_b_shuffled = np.random.permutation(train_idx_b)

    val_split_a = int(0.9 * len(train_idx_a_shuffled))
    val_split_b = int(0.9 * len(train_idx_b_shuffled))

    actual_train_idx_a = train_idx_a_shuffled[:val_split_a]
    actual_val_idx_a = train_idx_a_shuffled[val_split_a:]
    actual_train_idx_b = train_idx_b_shuffled[:val_split_b]
    actual_val_idx_b = train_idx_b_shuffled[val_split_b:]

    print(f"  Training: {len(actual_train_idx_a) + len(actual_train_idx_b):,} samples")
    print(f"  Validation: {len(actual_val_idx_a) + len(actual_val_idx_b):,} samples")
    print(f"  Test: {len(test_idx_a) + len(test_idx_b):,} samples")
    sys.stdout.flush()

    # INDEX INTO PRE-LOADED DATA
    print("[DEBUG] Indexing into pre-loaded data...")
    index_start = time.time()
    sys.stdout.flush()

    # Training data - directly index from combined arrays
    train_indices = np.concatenate([actual_train_idx_a, actual_train_idx_b + len(X_a)])
    X_train = X_combined[train_indices]
    y_train = labels_combined[train_indices]

    # Validation data
    val_indices = np.concatenate([actual_val_idx_a, actual_val_idx_b + len(X_a)])
    X_val = X_combined[val_indices]
    y_val = labels_combined[val_indices]

    # Test data
    test_indices = np.concatenate([test_idx_a, test_idx_b + len(X_a)])
    X_test = X_combined[test_indices]
    y_test = labels_combined[test_indices]

    index_elapsed = time.time() - index_start
    print(f"  ✓ Indexing complete in {index_elapsed:.3f} seconds")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"    Memory for fold: ~{(X_train.nbytes + X_val.nbytes + X_test.nbytes) / 1024**3:.1f} GB")
    sys.stdout.flush()

    # Create model
    print("[DEBUG] Creating model...")
    sys.stdout.flush()

    model = create_model(
        input_shape=input_shape,
        num_conv_layers=best_params['num_conv_layers'],
        filters=best_params['filters'],
        kernel_size=best_params['kernel_size'],
        pool_size=best_params['pool_size'],
        dropout_rate=best_params['dropout_rate'],
        dense_units=best_params['dense_units'],
        learning_rate=best_params['learning_rate'],
        lstm_units=best_params.get('lstm_units', 0)
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\nTraining model...")
    sys.stdout.flush()

    history = model.fit(
        X_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print(f"Training complete! Stopped at epoch {len(history.history['loss'])}")

    # Save training history for this fold
    history_dict = {
        'fold': fold_idx + 1,
        'subject': test_subject,
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'epochs_trained': len(history.history['loss'])
    }
    all_histories.append(history_dict)

    # Predict — both binary and probability
    y_prob = model.predict(X_test, batch_size=best_params['batch_size'], verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # Store results
    fold_results.append({
        'fold': fold_idx + 1,
        'subject': test_subject,
        'n_test_samples': len(y_test),
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    })

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)

    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    fold_elapsed = time.time() - fold_start
    total_elapsed = time.time() - total_start
    avg_per_fold = total_elapsed / (fold_idx + 1)
    remaining = avg_per_fold * (26 - fold_idx - 1)

    print(f"\nTiming:")
    print(f"  Fold time: {timedelta(seconds=int(fold_elapsed))}")
    print(f"  Total elapsed: {timedelta(seconds=int(total_elapsed))}")
    print(f"  Estimated remaining: {timedelta(seconds=int(remaining))}")
    sys.stdout.flush()

    # Save individual fold results
    wt_ref, data_len = get_subject_sensor_data(ref_metadata, test_subject, sensor_type="LowerBack")
    weartime_list_, total_weartime_samples_, total_weartime_seconds, total_weartime_minutes_, total_weartime_hours_, coverage = overlapping_windows_to_sample_labels(
        y_pred.tolist(), data_len)
    performance_metrics = wtd_performance_metrics(weartime_list_, wt_ref, data_len)

    output_dict = {
        "participant_id": test_subject,
        "position": "LowerBack",
        "algo": "CNN_LSTM_WTD",
        "version": "fullfeatures",
        "thresholds": {
            'num_conv_layers': best_params['num_conv_layers'],
            'kernel_size': best_params['kernel_size'],
            'pool_size': best_params['pool_size'],
            'dropout_rate': best_params['dropout_rate'],
            'dense_units': best_params['dense_units'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'epochs': best_params['epochs'],
            'lstm_units': best_params['lstm_units']
        },
        "total_file_length": data_len,
        "wt_sequences": [
            {
                "wt_id": i,
                "start": int(row["start"]),
                "end": int(row["end"])
            } for i, row in weartime_list_.iterrows()
        ],
        "total_weartime_samples": int(total_weartime_samples_),
        "total_weartime_hours": float(total_weartime_hours_),
        "total_weartime_minutes": float(total_weartime_minutes_),
        "performance_metrics": performance_metrics,
        "predictions": {
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        },
        "training_history": history_dict
    }

    # Save results
    if test_subject in IDs_a:
        save_dir = "/mnt/SFData/SUSTAIN_outputs/WTD/outputs_a/wearable/WTD/WTD_cnn_lstm"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{test_subject}__LowerBack.json")
        with open(filepath, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Saved: {filepath}")

    elif test_subject in IDs_b:
        save_dir = "/mnt/SFData/SUSTAIN_outputs/WTD/outputs_b/wearable/WTD/WTD_cnn_lstm"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{test_subject}__LowerBack.json")
        with open(filepath, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Saved: {filepath}")

    # Part A waking hour analysis
    if test_subject in IDs_a:
        performance_metrics = wtd_performance_metrics_waking_hours(weartime_list_, wt_ref, data_len)

        output_dict = {
            "participant_id": test_subject,
            "position": "LowerBack",
            "algo": "CNN_LSTM_WTD",
            "version": "fullfeatures",
            "thresholds": {
                'num_conv_layers': best_params['num_conv_layers'],
                'kernel_size': best_params['kernel_size'],
                'pool_size': best_params['pool_size'],
                'dropout_rate': best_params['dropout_rate'],
                'dense_units': best_params['dense_units'],
                'learning_rate': best_params['learning_rate'],
                'batch_size': best_params['batch_size'],
                'epochs': best_params['epochs'],
                'lstm_units': best_params['lstm_units']
            },
            "total_file_length": data_len,
            "wt_sequences": [
                {"wt_id": i, "start": int(row["start"]), "end": int(row["end"])}
                for i, row in weartime_list_.iterrows()
            ],
            "total_weartime_samples": int(total_weartime_samples_),
            "total_weartime_hours": float(total_weartime_hours_),
            "total_weartime_minutes": float(total_weartime_minutes_),
            "performance_metrics": performance_metrics,
            "predictions": {
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist()
            },
            "training_history": history_dict
        }

        save_dir = "/mnt/SFData/SUSTAIN_outputs/WTD/outputs_a_waking_hours/wearable/WTD/WTD_cnn_lstm"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{test_subject}__LowerBack.json")
        with open(filepath, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Saved waking hours: {filepath}")

    # Cleanup model and predictions (data stays in memory for next fold)
    del model, y_pred
    K.clear_session()
    gc.collect()

    print(f"{'=' * 60}\n")

# ── Aggregate results ─────────────────────────────────────────
total_time = time.time() - total_start

print("\n" + "=" * 60)
print("FINAL AGGREGATED RESULTS")
print("=" * 60)

# Check if we have any aggregated data (from either new or loaded folds)
if not all_y_test:
    print("ERROR: No fold data available for aggregation.")
    print("Check that fold JSONs exist and contain 'predictions' field.")
    print(f"\nTotal runtime: {timedelta(seconds=int(total_time))}")
    print("=" * 60)
else:
    # Compute metrics from aggregated data
    overall_cm = confusion_matrix(all_y_test, all_y_pred)

    # If we have per-fold metrics (from newly trained folds), display them
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"Mean Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
        print(f"Mean Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
        print(f"Mean F1-Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    else:
        # All folds were loaded - compute overall accuracy from aggregated predictions
        overall_acc = accuracy_score(all_y_test, all_y_pred)
        overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
            all_y_test, all_y_pred, average='binary', zero_division=0
        )
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall Precision: {overall_prec:.4f}")
        print(f"Overall Recall: {overall_rec:.4f}")
        print(f"Overall F1-Score: {overall_f1:.4f}")
        print("(Computed from aggregated predictions across all loaded folds)")

    print(f"\nOverall Confusion Matrix:\n{overall_cm}")
    print(f"\nTotal runtime: {timedelta(seconds=int(total_time))}")
    print("=" * 60)

    # Save results JSON (whether from new training or loaded folds)
    if fold_results:
        # We have per-fold metrics from newly trained folds
        results_df = pd.DataFrame(fold_results)
        output_data = {
            'model_type': 'CNN_LSTM',
            'hyperparameters': best_params,
            'mean_accuracy': float(results_df['accuracy'].mean()),
            'std_accuracy': float(results_df['accuracy'].std()),
            'mean_precision': float(results_df['precision'].mean()),
            'std_precision': float(results_df['precision'].std()),
            'mean_recall': float(results_df['recall'].mean()),
            'std_recall': float(results_df['recall'].std()),
            'mean_f1_score': float(results_df['f1_score'].mean()),
            'std_f1_score': float(results_df['f1_score'].std()),
            'overall_confusion_matrix': overall_cm.tolist(),
            'total_runtime_seconds': int(total_time),
            'per_fold_results': fold_results
        }
    else:
        # All folds were loaded - compute overall metrics
        overall_acc = accuracy_score(all_y_test, all_y_pred)
        overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
            all_y_test, all_y_pred, average='binary', zero_division=0
        )
        output_data = {
            'model_type': 'CNN_LSTM',
            'hyperparameters': best_params,
            'mean_accuracy': float(overall_acc),
            'std_accuracy': None,  # Not available when loading from individual JSONs
            'mean_precision': float(overall_prec),
            'std_precision': None,
            'mean_recall': float(overall_rec),
            'std_recall': None,
            'mean_f1_score': float(overall_f1),
            'std_f1_score': None,
            'overall_confusion_matrix': overall_cm.tolist(),
            'total_runtime_seconds': int(total_time),
            'per_fold_results': []  # Not available when all folds were loaded
        }

    output_file = Path(__file__).parent / "CNN_LSTM_results_lowback.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save training histories JSON
    if all_histories:
        history_file = Path(__file__).parent / "CNN_LSTM_training_histories_lowback.json"
        with open(history_file, 'w') as f:
            json.dump(all_histories, f, indent=2)
        print(f"Training histories saved to {history_file}")

    # ── Generate plots ────────────────────────────────────────
    all_y_true = np.array(all_y_test)
    all_y_prob_arr = np.array(all_y_prob)
    plot_dir = Path(__file__).parent.parent / "figure_extraction_training"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # [Rest of plotting code remains the same - lines 464-554]
    # 1. Calibration Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    fraction_pos, mean_pred = calibration_curve(all_y_true, all_y_prob_arr, n_bins=10)
    ax.plot(mean_pred, fraction_pos, 'o-', color='steelblue', label='CNN_LSTM')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "cnn_lstm_lowback_calibration_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    precision_vals, recall_vals, _ = precision_recall_curve(all_y_true, all_y_prob_arr)
    ap = average_precision_score(all_y_true, all_y_prob_arr)
    ax.plot(recall_vals, precision_vals, color='steelblue', lw=2, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "cnn_lstm_lowback_precision_recall_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob_arr)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "cnn_lstm_lowback_roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Loss curves (mean ± std across folds)
    if all_histories:
        max_epochs = max(h['epochs_trained'] for h in all_histories)


        def pad_history(values, max_len):
            return values + [values[-1]] * (max_len - len(values))


        train_loss = np.array([pad_history(h['loss'], max_epochs) for h in all_histories])
        val_loss = np.array([pad_history(h['val_loss'], max_epochs) for h in all_histories])
        train_acc = np.array([pad_history(h['accuracy'], max_epochs) for h in all_histories])
        val_acc = np.array([pad_history(h['val_accuracy'], max_epochs) for h in all_histories])
        epochs = np.arange(1, max_epochs + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss.mean(axis=0), color='steelblue', lw=2, label='Training loss')
        ax.fill_between(epochs,
                        train_loss.mean(axis=0) - train_loss.std(axis=0),
                        train_loss.mean(axis=0) + train_loss.std(axis=0),
                        alpha=0.15, color='steelblue')
        ax.plot(epochs, val_loss.mean(axis=0), color='tomato', lw=2, label='Validation loss')
        ax.fill_between(epochs,
                        val_loss.mean(axis=0) - val_loss.std(axis=0),
                        val_loss.mean(axis=0) + val_loss.std(axis=0),
                        alpha=0.15, color='tomato')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "cnn_lstm_lowback_loss_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 5. Accuracy curves (mean ± std across folds)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_acc.mean(axis=0), color='steelblue', lw=2, label='Training accuracy')
        ax.fill_between(epochs,
                        train_acc.mean(axis=0) - train_acc.std(axis=0),
                        train_acc.mean(axis=0) + train_acc.std(axis=0),
                        alpha=0.15, color='steelblue')
        ax.plot(epochs, val_acc.mean(axis=0), color='tomato', lw=2, label='Validation accuracy')
        ax.fill_between(epochs,
                        val_acc.mean(axis=0) - val_acc.std(axis=0),
                        val_acc.mean(axis=0) + val_acc.std(axis=0),
                        alpha=0.15, color='tomato')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "cnn_lstm_lowback_accuracy_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("\nPlots saved:")
    print("  cnn_lstm_lowback_calibration_curve.png")
    print("  cnn_lstm_lowback_precision_recall_curve.png")
    print("  cnn_lstm_lowback_roc_curve.png")
    print("  cnn_lstm_lowback_loss_curves.png")
    print("  cnn_lstm_lowback_accuracy_curves.png")