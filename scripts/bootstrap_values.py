import numpy as np
import pandas as pd
import torch
import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import math
from eval import evaluate_internal

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold that maximizes the F1 score.
    """
    best_threshold = 0.5
    best_f1 = 0

    # Iterate over potential thresholds
    thresholds = np.unique(probabilities)

    # Downsample thresholds if there are too many (optional, for speed)
    if len(thresholds) > 1000:
        thresholds = np.linspace(0, 1, 101)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(true_labels, predictions, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold

# Path to inference output directory
data_dir = "./fixedloss_16k_newinference/"

# Load predicted and labels data
predicted_data = np.load(Path(data_dir) / 'predicted_weights.npz')
labels_data = np.load(Path(data_dir) / 'labels_weights.npz')

labels = labels_data['data']
predicted = predicted_data['data']

# ==============================================================================
# APPLY SIGMOID HERE
# ==============================================================================
# This applies the function 1 / (1 + exp(-x)) to every single element independently.
# It does NOT make the row sum to 1. It keeps classes independent.
predicted_tensor = torch.from_numpy(predicted)

temperature = 14.0
predicted_tensor = predicted_tensor / temperature

predicted_tensor = torch.sigmoid(predicted_tensor)

predicted = predicted_tensor.numpy()
# ==============================================================================
print(predicted)
# Thresholds list
thresholds = []

# Find threshold for each label
for i in range(18):
    prob = predicted[:, i]
    l = labels[:, i]
    threshold = find_threshold(prob, l)
    thresholds.append(threshold)

print("Calculated Thresholds:", thresholds)

# Initialize DataFrames for storing evaluation metrics
concatenated_df_auroc = pd.DataFrame()
concatenated_df_f1 = pd.DataFrame()
concatenated_df_acc = pd.DataFrame()
concatenated_df_precision = pd.DataFrame()

pathologies = ['Medical material', 'Calcification', 'Cardiomegaly', 'Pericardial effusion',
               'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy',
               'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
               'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
               'Bronchiectasis', 'Interlobular septal thickening']

# Bootstrap iterations
for _ in tqdm.tqdm(range(100)):
    # Standard bootstrapping: sample with replacement
    indices = np.random.choice(range(len(labels)), size=len(labels), replace=True)
    sampled_labels = labels[indices]
    sampled_predicted = predicted[indices]

    # Evaluate internal metrics (AUROC)
    dfs_auroc = evaluate_internal(sampled_predicted, sampled_labels, pathologies, data_dir)
    concatenated_df_auroc = pd.concat([concatenated_df_auroc, dfs_auroc])

    f1s = []
    accs = []
    precisions = []

    # Calculate metrics for each label using the pre-calculated thresholds
    for i in range(18):
        prob = sampled_predicted[:, i]
        label = sampled_labels[:, i]
        threshold = thresholds[i]

        # Apply threshold
        pred = (prob > threshold).astype(int)

        f1s.append(f1_score(label, pred, zero_division=0))
        accs.append(accuracy_score(label, pred))
        precisions.append(precision_score(label, pred, zero_division=0))

    # Store metrics in DataFrames
    concatenated_df_f1 = pd.concat([concatenated_df_f1, pd.DataFrame([f1s], columns=pathologies)])
    concatenated_df_acc = pd.concat([concatenated_df_acc, pd.DataFrame([accs], columns=pathologies)])
    concatenated_df_precision = pd.concat([concatenated_df_precision, pd.DataFrame([precisions], columns=pathologies)])

# Save results
writer = pd.ExcelWriter(Path(data_dir) / 'aurocs_bootstrap.xlsx', engine='xlsxwriter')
concatenated_df_auroc.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()

writer = pd.ExcelWriter(Path(data_dir) / 'f1_bootstrap.xlsx', engine='xlsxwriter')
concatenated_df_f1.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()

writer = pd.ExcelWriter(Path(data_dir) / 'acc_bootstrap.xlsx', engine='xlsxwriter')
concatenated_df_acc.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()

writer = pd.ExcelWriter(Path(data_dir) / 'precision_bootstrap.xlsx', engine='xlsxwriter')
concatenated_df_precision.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()