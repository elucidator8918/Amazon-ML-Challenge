import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Read the CSV file
df = pd.read_csv('train_f1_processed.csv')

# Initialize counts
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives
TN = 0  # True Negatives

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    GT = row['entity_value']       # Ground truth value (GT)
    OUT = row['extracted_value']   # Output value (OUT)
    
    # Check if GT and OUT are NaN
    GT_is_nan = pd.isna(GT)
    OUT_is_nan = pd.isna(OUT)
    
    # Apply the evaluation criteria
    if not GT_is_nan and not OUT_is_nan:
        if OUT == GT:
            TP += 1  # True Positive
        else:
            FP += 1  # False Positive (OUT != GT)
    elif GT_is_nan and OUT_is_nan:
        TN += 1  # True Negative (Both OUT and GT are empty)
    elif not GT_is_nan and OUT_is_nan:
        FN += 1  # False Negative (No OUT but GT exists)
    elif GT_is_nan and not OUT_is_nan:
        FP += 1  # False Positive (OUT provided but GT is empty)

# Print the counts
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): {TN}")

# Calculate Precision, Recall, and F1 Score
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1_score = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")
print(f"F1 Score: {F1_score:.4f}")

# Prepare the confusion matrix using the counts
# Confusion matrix format:
#               Predicted
#               Negative    Positive
# Actual Negative    TN          FP
# Actual Positive    FN          TP

cm = np.array([[TN, FP],
               [FN, TP]])

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
