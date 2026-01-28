# FINAL SUBMISSION

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 1. Load Data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# 2. Combine for Feature Engineering
train['is_train'] = 1
test['is_train'] = 0
# Rename columns to match if necessary (based on your snippet, names look consistent)
df = pd.concat([train, test], ignore_index=True)

# 3. Feature Engineering
# Time features
df['hour'] = df['step'] % 24
df['day'] = (df['step'] // 24).astype(int)

# Merchant Flag (Strong predictor: Merchants are usually not fraud sources in this dataset)
df['is_merchant'] = df['nameDest'].str.startswith('M').astype(int)

# Frequency Encoding
for col in ['nameDest', 'nameOrig']:
    df[col + '_count'] = df.groupby(col)['id'].transform('count')

# Amount features
df['log_amount'] = np.log1p(df['amount'])  # Log transform helps with skewed amount data
df['amount_decimal'] = df['amount'] % 1

# Interaction: Amount relative to the destination's average
# (Handling cases where destination appears only once)
dest_means = df.groupby('nameDest')['amount'].transform('mean')
df['amount_to_mean_dest'] = df['amount'] / dest_means

# 4. Prepare Data
features = ['step', 'amount', 'log_amount', 'hour', 'day', 'is_merchant', 
            'nameDest_count', 'nameOrig_count', 'amount_decimal', 
            'amount_to_mean_dest', 'isFlaggedFraud']

X = df[df['is_train'] == 1][features]
y = df[df['is_train'] == 1]['isFraud']
X_test = df[df['is_train'] == 0][features]
test_ids = df[df['is_train'] == 0]['id']

# Handle NaNs (HistGradientBoosting handles them, but good to be safe with engineered feats)
X = X.fillna(-1)
X_test = X_test.fillna(-1)

# 5. Cross-Validation Ensemble
# Using 10 folds reduces variance and usually improves Log Loss score
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Array to store test predictions
test_preds_accumulator = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

print("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Tuned Hyperparameters: Lower learning rate + higher iterations = better generalization
    model = HistGradientBoostingClassifier(
        learning_rate=0.01,        # Slower learning for better precision
        max_iter=2000,             # More trees to compensate for low learning rate
        max_depth=12,              # Slightly deeper trees
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,       # Patience for early stopping
        random_state=42 + fold,    # Change seed slightly per fold
        scoring='neg_log_loss'
    )
    
    model.fit(X_train_fold, y_train_fold)
    
    # Predict on test set for this fold
    fold_preds = model.predict_proba(X_test)[:, 1]
    test_preds_accumulator += fold_preds
    
    print(f"Fold {fold+1} completed.")

# Average the predictions across all folds
final_predictions = test_preds_accumulator / kf.get_n_splits()

# 6. Post-Processing
# Merchant Rule: Merchants are consistently safe in this specific dataset
# We apply this mask to the averaged predictions
is_merchant_mask = X_test['is_merchant'] == 1
final_predictions[is_merchant_mask] = 0.0

# Flagged Rule: Softened to 0.99 instead of 0.999
# Log Loss penalizes being "wrong and confident" heavily. 
# 0.99 protects you from massive penalties if there's one outlier.
is_flagged_mask = X_test['isFlaggedFraud'] == 1
final_predictions[is_flagged_mask] = np.maximum(final_predictions[is_flagged_mask], 0.99)

# Safety Clip: 1e-6 is standard to prevent log(0)
final_predictions = np.clip(final_predictions, 1e-6, 1 - 1e-6)

# 7. Save
submission = pd.DataFrame({'id': test_ids, 'prediction': final_predictions})
submission.to_csv('submission_optimized.csv', index=False)

print("Submission saved!")






'''
# LATE SUBMISSION (had a 4 * 10^-4 difference in log loss from the first submission, very minor difference)

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# 2. Combine for Feature Engineering
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)

# 3. Create Features
# Time features
df['hour'] = df['step'] % 24
df['day'] = (df['step'] // 24).astype(int)

# Merchant Flag (Crucial for the 0.0 rule)
df['is_merchant'] = df['nameDest'].str.startswith('M').astype(int)

# Frequency Encoding
for col in ['nameDest', 'nameOrig']:
    df[col + '_count'] = df.groupby(col)['id'].transform('count')

# Amount Interaction
df['amount_to_mean_dest'] = df['amount'] / df.groupby('nameDest')['amount'].transform('mean')
df['amount_decimal'] = df['amount'] % 1

# Handle Missing Values (created by division)
df.fillna(-1, inplace=True)

# 4. Prepare for Training
features = ['step', 'amount', 'hour', 'day', 'is_merchant', 
            'nameDest_count', 'nameOrig_count', 'amount_decimal', 
            'amount_to_mean_dest', 'isFlaggedFraud']

X = df[df['is_train'] == 1][features]
y = df[df['is_train'] == 1]['isFraud']
X_test = df[df['is_train'] == 0][features]

# 5. Train Model (HistGradientBoosting is faster & more accurate here)
print("Training Model...")
model = HistGradientBoostingClassifier(
    learning_rate=0.05, 
    max_iter=500, 
    max_depth=10, 
    l2_regularization=1.0, 
    random_state=42
)
model.fit(X, y)

# 6. Generate Predictions
predictions = model.predict_proba(X_test)[:, 1]

# 7. Post-Processing (The "Magic" Fixes)

# RULE 1: Merchants are NEVER fraud -> Force Exact 0.0
# This fixes the "no 0 is self" issue you mentioned.
is_merchant_test = X_test['is_merchant'] == 1
predictions[is_merchant_test] = 0.0

# RULE 2: Flagged transactions are ALWAYS fraud -> Force 0.999
predictions[X_test['isFlaggedFraud'] == 1] = 0.999

# RULE 3: Safety Clip (for non-merchants only)
# We clip others to avoid log_loss penalties, but keep 0.0 for merchants
predictions[~is_merchant_test] = np.clip(predictions[~is_merchant_test], 1e-6, 1-1e-6)

# 8. Save Submission
submission = pd.DataFrame({'id': test['id'], 'prediction': predictions})
submission.to_csv('submission_exact_zeros.csv', index=False)

print("Done! Saved 'submission_exact_zeros.csv'")
print(f"Merchant Rows (set to 0.0): {is_merchant_test.sum()}")
print(f"Final Mean Prediction: {submission['prediction'].mean()}")
'''