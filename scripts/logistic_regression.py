import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CS32h/sample_data.csv")
df_model = df.dropna(subset=['isPositive']).copy()
df_model['isPositive'] = df_model['isPositive'].astype(int)
df_model['combined_text'] = df_model['user_query'].fillna('') + ' ' + df_model['chatehr_response'].fillna('')

categorical_features = ['prov_type', 'model', 'subcategory', 'nlp_task', 'category', 'task_name', 'department_name']
numeric_features = ['prompt_tokens']
target = 'isPositive'

# Features for each experiment
X_qrc = df_model[categorical_features + ['combined_text'] + numeric_features]
X_qr = df_model[['combined_text']]
X_qc = df_model[categorical_features + ['user_query'] + numeric_features]
X_q = df_model[['user_query']]
X_c = df_model[categorical_features + numeric_features]

y = df_model[target]


preprocessor_qrc = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=3072,ngram_range=(1,2) ), 'combined_text')
    ],
    remainder='passthrough',
    sparse_threshold=0
)

preprocessor_qr = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=3072, ngram_range=(1,2)), 'combined_text')
    ],
    remainder='passthrough',
    sparse_threshold=0
)

preprocessor_qc = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=3072, ngram_range=(1,2)), 'user_query')
    ],
    remainder='passthrough',
    sparse_threshold=0
)

preprocessor_q = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=3072, ngram_range=(1,2)), 'user_query')
    ],
    remainder='passthrough',
    sparse_threshold=0
)

preprocessor_c = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough',
    sparse_threshold=0
)


clf_qrc = Pipeline([
    ('preprocessor', preprocessor_qrc),
    ('classifier', LogisticRegression(max_iter=5000, class_weight="balanced"))
])

clf_qr = Pipeline([
    ('preprocessor', preprocessor_qr),
    ('classifier', LogisticRegression(max_iter=5000, class_weight="balanced"))
])

clf_qc = Pipeline([
    ('preprocessor', preprocessor_qc),
    ('classifier', LogisticRegression(max_iter=5000, class_weight="balanced"))
])

clf_q = Pipeline([
    ('preprocessor', preprocessor_q),
    ('classifier', LogisticRegression(max_iter=5000, class_weight="balanced"))
])

clf_c = Pipeline([
    ('preprocessor', preprocessor_c),
    ('classifier', LogisticRegression(max_iter=5000, class_weight="balanced"))
])

def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    Lower is better (0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def top_confidence_accuracy(y_true, y_pred_proba, top_pct=0.25):
    """
    Calculate accuracy on the top 25% most confident predictions for each class.
    
    Returns:
        dict with accuracy for top 25% confident accepts and rejects
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Compute confidence (distance from decision boundary)
    confidence = np.abs(y_pred_proba - 0.5)
    
    # Determine number of top samples
    n_top = max(1, int(len(y_true) * top_pct))
    top_indices = np.argsort(confidence)[-n_top:]  # highest confidence
    
    # Filter top confident subset
    y_true_top = y_true[top_indices]
    y_pred_top = (y_pred_proba[top_indices] >= 0.5).astype(int)
    
    # Separate by true class
    accept_mask = (y_true_top == 1)
    reject_mask = (y_true_top == 0)
    
    acc_accept = accuracy_score(y_true_top[accept_mask], y_pred_top[accept_mask]) if accept_mask.any() else np.nan
    acc_reject = accuracy_score(y_true_top[reject_mask], y_pred_top[reject_mask]) if reject_mask.any() else np.nan
    
    return {
        "acc_top_accept": acc_accept,
        "acc_top_reject": acc_reject,
        "n_top_total": n_top,
        "n_accept": accept_mask.sum(),
        "n_reject": reject_mask.sum()
    }

def evaluate_model_kfold_with_calibration(clf, X, y, label, cv=5):
    """
    K-fold cross-validation with calibration and uncertainty metrics
    """
    print(f"\n{'='*80}")
    print(f"K-Fold Cross-Validation for {label} (k={cv})")
    print(f"{'='*80}")
    
    # Standard scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Run cross-validation
    cv_results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Manual k-fold for calibration metrics
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    calibration_metrics = {
        'log_loss': [],
        'ece': [],
        'acc_top25_accept': [],
        'acc_top25_reject': []
    }
    
    all_y_true = []
    all_y_pred_proba = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model
        clf_fold = Pipeline([('preprocessor', clf.named_steps['preprocessor']),
                            ('classifier', clf.named_steps['classifier'])])
        clf_fold.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = clf_fold.predict_proba(X_val)[:, 1]
        
        # Store for overall metrics
        all_y_true.extend(y_val.values)
        all_y_pred_proba.extend(y_pred_proba)
        
        # Calculate calibration metrics
        calibration_metrics['log_loss'].append(log_loss(y_val, y_pred_proba))
        calibration_metrics['ece'].append(expected_calibration_error(y_val.values, y_pred_proba))
        
        # Calculate top 25% confidence accuracy
        top_conf_metrics = top_confidence_accuracy(y_val.values, y_pred_proba)
        calibration_metrics['acc_top25_accept'].append(top_conf_metrics['acc_top_accept'])
        calibration_metrics['acc_top25_reject'].append(top_conf_metrics['acc_top_reject'])
    
    # Print standard metrics
    print(f"\n{'─'*80}")
    print("STANDARD METRICS")
    print(f"{'─'*80}")
    print(f"\nTest Set Metrics (Mean ± Std across {cv} folds):")
    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
    print(f"  F1 Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"  AUROC:     {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    
    # Print calibration metrics
    print(f"\n{'─'*80}")
    print("CALIBRATION METRICS (Lower is Better)")
    print(f"{'─'*80}")
    print(f"  Log Loss:    {np.mean(calibration_metrics['log_loss']):.4f} ± {np.std(calibration_metrics['log_loss']):.4f}")
    print(f"  ECE (Expected): {np.mean(calibration_metrics['ece']):.4f} ± {np.std(calibration_metrics['ece']):.4f}")
    
    # Print top 25% confidence accuracy
    print(f"\n{'─'*80}")
    print("TOP 25% CONFIDENCE ACCURACY (Higher is Better)")
    print(f"{'─'*80}")
    print(f"  Acc (Top 25% Accept): {np.mean(calibration_metrics['acc_top25_accept']):.4f} ± {np.std(calibration_metrics['acc_top25_accept']):.4f}")
    print(f"  Acc (Top 25% Reject): {np.mean(calibration_metrics['acc_top25_reject']):.4f} ± {np.std(calibration_metrics['acc_top25_reject']):.4f}")
    
    return {
        'cv_results': cv_results,
        'calibration_metrics': calibration_metrics,
        'all_y_true': np.array(all_y_true),
        'all_y_pred_proba': np.array(all_y_pred_proba)
    }

experiments = {
    'QRC': (X_qrc, clf_qrc),
    'QR': (X_qr, clf_qr),
    'QC': (X_qc, clf_qc),
    'Q': (X_q, clf_q),
    'C': (X_c, clf_c)
}
results = {}
for label, (X, clf) in experiments.items():
    results[label] = evaluate_model_kfold_with_calibration(clf, X, y, label, cv=5)

print("\n" + "="*80)
print("SUMMARY: Comprehensive Model Comparison")
print("="*80)

summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AUROC': [results[label]['cv_results']['test_roc_auc'].mean() for label in results.keys()],
    'AUROC_std': [results[label]['cv_results']['test_roc_auc'].std() for label in results.keys()],
    'ECE': [np.mean(results[label]['calibration_metrics']['ece']) for label in results.keys()],
    'Acc_Top25_Accept': [np.mean(results[label]['calibration_metrics']['acc_top25_accept']) for label in results.keys()],
    'Acc_Top25_Reject': [np.mean(results[label]['calibration_metrics']['acc_top25_reject']) for label in results.keys()]
})

summary_df = summary_df.sort_values('AUROC', ascending=False)
print("\n" + summary_df.to_string(index=False))

def get_feature_importance(clf, X, y, label, top_n=10):
    """
    Extract and display top N most important features from trained model
    """
    print(f"\n{'='*80}")
    print(f"Top {top_n} Most Informative Features for {label}")
    print(f"{'='*80}")
    
    # Train the full model on all data to get feature importances
    clf_full = Pipeline([('preprocessor', clf.named_steps['preprocessor']),
                        ('classifier', clf.named_steps['classifier'])])
    clf_full.fit(X, y)
    
    # Get feature names from preprocessor
    feature_names = []
    
    # Extract feature names from each transformer
    transformers = clf_full.named_steps['preprocessor'].transformers_
    
    for name, transformer, columns in transformers:
        if name == 'cat':  # Categorical features (OneHotEncoder)
            cat_features = transformer.get_feature_names_out(columns)
            feature_names.extend(cat_features)
        elif name == 'text':  # Text features (TfidfVectorizer)
            vocab = transformer.get_feature_names_out()
            # Use the column name directly (it's a string, not a list)
            col_name = columns if isinstance(columns, str) else columns[0]
            feature_names.extend([f"{col_name}__{word}" for word in vocab])
        elif name == 'remainder':  # Numeric features passed through
            # Get the remainder columns
            remainder_cols = clf_full.named_steps['preprocessor']._remainder[2]
            if isinstance(remainder_cols, list):
                feature_names.extend(remainder_cols)
            elif hasattr(remainder_cols, 'tolist'):
                feature_names.extend(remainder_cols.tolist())
    
    # Get coefficients from logistic regression
    coefficients = clf_full.named_steps['classifier'].coef_[0]
    
    # Ensure lengths match
    if len(feature_names) != len(coefficients):
        print(f"WARNING: Feature names ({len(feature_names)}) != Coefficients ({len(coefficients)})")
        # Truncate or pad feature names to match coefficients
        if len(feature_names) > len(coefficients):
            feature_names = feature_names[:len(coefficients)]
        else:
            # Add generic names for missing features
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), len(coefficients))])
    
    # Create dataframe of features and coefficients
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Sort by absolute coefficient value
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Display results
    print(f"\nFeature Name                                      | Coefficient | Impact")
    print(f"{'-'*80}")
    for idx, row in top_features.iterrows():
        impact = "→ POSITIVE" if row['coefficient'] > 0 else "→ NEGATIVE"
        print(f"{row['feature'][:48]:48} | {row['coefficient']:11.4f} | {impact}")
    
    return feature_importance

# Extract feature importance for all models
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importances = {}
for label, (X, clf) in experiments.items():
    feature_importances[label] = get_feature_importance(clf, X, y, label, top_n=10)
