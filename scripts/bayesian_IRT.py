import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score, precision_score, 
                             recall_score, f1_score)
from sklearn.calibration import calibration_curve
import pymc as pm
import arviz as az

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("CS32h/sample_data.csv")

class BayesianIRT:
        """
        Bayesian hierarchical logistic regression with IRT parameters
        Now includes TF-IDF features and k-fold cross validation
        """
        
        def __init__(self, df, n_folds=4, random_state=42,
                     use_tfidf=True, tfidf_max_features=3072, tfidf_ngram_range=(1, 1)):
            self.df = df.copy()
            self.n_folds = n_folds
            self.random_state = random_state
            
            # TF-IDF parameters
            self.use_tfidf = use_tfidf
            self.tfidf_max_features = tfidf_max_features
            self.tfidf_ngram_range = tfidf_ngram_range
            
            # Storage for k-fold results
            self.fold_results = []
            
        def prepare_data(self):
            """Prepare data for Bayesian modeling with TF-IDF and k-fold"""
            
            self.df['accepted'] = self.df['isPositive'].astype(int)
            
            # Create combined text
            print("Creating combined text features...")
            self.df['combined_text'] = (
                self.df['user_query'].fillna('') + ' ' + 
                self.df['chatehr_response'].fillna('')
            )
            
            print("\n" + "="*80)
            print("BAYESIAN MODEL - K-FOLD CROSS VALIDATION SETUP")
            print("="*80)
            print(f"Total samples: {len(self.df)}")
            print(f"Number of folds: {self.n_folds}")
            
            # Initialize k-fold
            self.skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            # Initialize arrays
            self.df['pred_prob_mean'] = np.nan
            self.df['pred_prob_lower'] = np.nan
            self.df['pred_prob_upper'] = np.nan
            self.df['fold'] = -1
            
            return self
        
        def _prepare_fold_data(self, train_df, val_df):
            """Prepare data for a single fold"""
            
            # Categorical encoding
            encoders = {}
            encoded_cols = []
            
            categorical_features = ['prov_type', 'nlp_task', 'subcategory', 
                                   'category', 'task_name', 'model', 'department_name']
            
            for col in categorical_features:
                if col in train_df.columns:
                    unique_vals = train_df[col].fillna('missing').unique()
                    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                    
                    train_df[f'{col}_idx'] = train_df[col].fillna('missing').map(val_to_idx)
                    val_df[f'{col}_idx'] = val_df[col].fillna('missing').map(val_to_idx)
                    val_df[f'{col}_idx'] = val_df[f'{col}_idx'].fillna(-1).astype(int)
                    
                    encoders[col] = val_to_idx
                    encoded_cols.append(f'{col}_idx')

                
            train_df['prompt_tokens_scaled'] = train_df[['prompt_tokens']]
            val_df['prompt_tokens_scaled'] = val_df[['prompt_tokens']]
            # TF-IDF features
            n_tfidf_features = 0
            if self.use_tfidf:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.tfidf_max_features,
                    ngram_range=self.tfidf_ngram_range,
                    min_df=2,
                    max_df=0.95,
                    stop_words='english',
                    strip_accents='unicode',
                    lowercase=True
                )
                
                # Fit on training
                train_text = train_df['combined_text'].fillna('')
                tfidf_vectorizer.fit(train_text)
                
                # Transform both
                tfidf_train = tfidf_vectorizer.transform(train_text).toarray()
                val_text = val_df['combined_text'].fillna('')
                tfidf_val = tfidf_vectorizer.transform(val_text).toarray()
                
                # Add to dataframes
                for i in range(tfidf_train.shape[1]):
                    train_df[f'tfidf_{i}'] = tfidf_train[:, i]
                    val_df[f'tfidf_{i}'] = tfidf_val[:, i]
                
                n_tfidf_features = tfidf_train.shape[1]
            
            return train_df, val_df, encoders, encoded_cols, n_tfidf_features
        
        def fit_model(self, draws=2000, tune=1000, chains=4, balance_classes=False):
            """
            Fit Bayesian hierarchical model with k-fold cross validation
            """
            
            print("\n" + "="*80)
            print("FITTING BAYESIAN HIERARCHICAL MODEL WITH K-FOLD CV")
            print("="*80)
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            
            if avail_gb < 2.0:
                print(f"\n❌ INSUFFICIENT MEMORY: Only {avail_gb:.1f} GB available")
                print("Reducing to minimal configuration...")
                draws = 250
                tune = 250
                chains = 2
            
            # REDUCE COMPLEXITY IF NEEDED
            if self.tfidf_max_features > 10 and avail_gb < 4.0:
                print(f"⚠️  Reducing TF-IDF features: {self.n_tfidf_features} → 10")
                self.tfidf_max_features = 10
                self.prepare_data()  # Re-run with fewer features
            
            print(f"\nMemory available: {avail_gb:.1f} GB")
            print(f"Running: {draws} draws, {tune} tune, {chains} chains\n")
            
            fold_aucs = []
            
            for fold, (train_idx, val_idx) in enumerate(self.skf.split(self.df, self.df['accepted']), 1):
                print(f"\n{'='*80}")
                print(f"FOLD {fold}/{self.n_folds}")
                print(f"{'='*80}")

                # Split data
                train_df = self.df.iloc[train_idx].copy()
                val_df = self.df.iloc[val_idx].copy()
                
                print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
                
                # Prepare fold data
                train_df, val_df, encoders, encoded_cols, n_tfidf_features = self._prepare_fold_data(train_df, val_df)
                
                if fold == 1 and self.use_tfidf:
                    print(f"Created {n_tfidf_features} TF-IDF features")
                
                # Class weights
                if balance_classes:
                    n_samples = len(train_df)
                    n_accepted = train_df['accepted'].sum()
                    n_rejected = n_samples - n_accepted
                    
                    weight_accepted = n_samples / (2 * n_accepted)
                    weight_rejected = n_samples / (2 * n_rejected)
                    
                    weights = np.where(
                        train_df['accepted'].values == 1,
                        weight_accepted,
                        weight_rejected
                    )
                else:
                    weights = np.ones(len(train_df))
                
                # Build and fit model
                with pm.Model() as model:
                    
                    hierarchical_effects = {}
                    
                    # Hierarchical priors for categorical features
                    for col in ['department_name', 'prov_type', 'nlp_task', 
                               'model', 'category', 'subcategory', 'task_name']:
                        
                        col_idx = f'{col}_idx'
                        if col_idx in encoded_cols:
                            n_categories = len(encoders[col])
                            
                            mu = pm.Normal(f'{col}_mu', mu=0, sigma=1)
                            sigma = pm.HalfNormal(f'{col}_sigma', sigma=1)
                            
                            effects_offset = pm.Normal(
                                f'{col}_effects_offset',  
                                mu=0,
                                sigma=1,
                                shape=n_categories
                            )
                            

                            effects = pm.Deterministic(
                                f'{col}_effects',  
                                mu + sigma * effects_offset
                            )
                            
                            hierarchical_effects[col] = effects
                    
                    # Prior for prompt tokens
                    beta_tokens = pm.Normal('beta_tokens', mu=0, sigma=1)
                    
                    # Priors for TF-IDF features
                    if self.use_tfidf and n_tfidf_features > 0:
                        tfidf_sigma = pm.HalfNormal('tfidf_sigma', sigma=0.5)
                        beta_tfidf = pm.Normal(
                            'beta_tfidf',
                            mu=0,
                            sigma=tfidf_sigma,
                            shape=n_tfidf_features
                        )
                    
                    # Intercept
                    intercept = pm.Normal('intercept', mu=0, sigma=2)
                    
                    # Build linear predictor
                    logit_p = intercept + beta_tokens * train_df['prompt_tokens_scaled'].values
                    
                    # Add categorical effects
                    for col, effects in hierarchical_effects.items():
                        col_idx = f'{col}_idx'
                        indices = train_df[col_idx].values
                        logit_p = logit_p + effects[indices]
                    
                    # Add TF-IDF effects
                    if self.use_tfidf and n_tfidf_features > 0:
                        tfidf_matrix = np.column_stack([
                            train_df[f'tfidf_{i}'].values 
                            for i in range(n_tfidf_features)
                        ])
                        logit_p = logit_p + pm.math.dot(tfidf_matrix, beta_tfidf)
                    
                    # Likelihood
                    if balance_classes:
                        p = pm.math.sigmoid(logit_p)
                        log_likelihood = weights * (
                            train_df['accepted'].values * pm.math.log(p) +
                            (1 - train_df['accepted'].values) * pm.math.log(1 - p)
                        )
                        pm.Potential('weighted_obs', log_likelihood.sum())
                    else:
                        y_obs = pm.Bernoulli('y_obs', logit_p=logit_p,
                                            observed=train_df['accepted'].values)
                    
                    print(f"\nSampling from posterior (fold {fold})...")
                    trace = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        random_seed=self.random_state,
                        return_inferencedata=True,
                        progressbar=True,
                        # target_accept=0.85
                    )
                
                # Make predictions on validation set
                val_preds = self._predict_fold(trace, val_df, encoders, encoded_cols, n_tfidf_features)
                
                # Store predictions
                self.df.loc[val_idx, 'pred_prob_mean'] = val_preds['mean']
                self.df.loc[val_idx, 'pred_prob_lower'] = val_preds['lower_95']
                self.df.loc[val_idx, 'pred_prob_upper'] = val_preds['upper_95']
                self.df.loc[val_idx, 'fold'] = fold
                
                # Calculate metrics
                val_auc = roc_auc_score(val_df['accepted'], val_preds['mean'])
                fold_aucs.append(val_auc)
                
                print(f"\nFold {fold} Validation AUC: {val_auc:.4f}")
                
                # Store fold results
                self.fold_results.append({
                    'fold': fold,
                    'trace': trace,
                    'val_idx': val_idx,
                    'val_auc': val_auc,
                    'encoders': encoders,
                    'encoded_cols': encoded_cols,
                    'n_tfidf_features': n_tfidf_features
                })
            
            # Overall results
            print("\n" + "="*80)
            print("K-FOLD CROSS VALIDATION RESULTS")
            print("="*80)
            print(f"\nMean AUC across folds: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
            print(f"Fold-wise AUCs: {[f'{auc:.4f}' for auc in fold_aucs]}")
            
            return self
        
        def _predict_fold(self, trace, val_df, encoders, encoded_cols, n_tfidf_features):
            """Make predictions for a single fold"""
            
            intercept_samples = trace.posterior['intercept'].values.flatten()
            beta_tokens_samples = trace.posterior['beta_tokens'].values.flatten()
            
            n_samples = len(intercept_samples)
            n_val = len(val_df)
            
            predictions = np.zeros((n_samples, n_val))
            
            # Get categorical effects
            categorical_samples = {}
            for col in ['department_name', 'prov_type', 'nlp_task', 
                       'model', 'category', 'subcategory', 'task_name']:
                col_idx = f'{col}_idx'
                if col_idx in encoded_cols:
                    effects = trace.posterior[f'{col}_effects'].values
                    n_chains, n_draws, n_cats = effects.shape
                    effects_flat = effects.reshape(n_chains * n_draws, n_cats)
                    categorical_samples[col] = effects_flat
            
            # Get TF-IDF effects
            if self.use_tfidf and n_tfidf_features > 0:
                beta_tfidf_samples = trace.posterior['beta_tfidf'].values
                beta_tfidf_flat = beta_tfidf_samples.reshape(-1, n_tfidf_features)
                
                tfidf_matrix = np.column_stack([
                    val_df[f'tfidf_{i}'].values 
                    for i in range(n_tfidf_features)
                ])
            
            # Generate predictions
            for i in range(n_samples):
                logit_p = intercept_samples[i] + \
                         beta_tokens_samples[i] * val_df['prompt_tokens_scaled'].values
                
                # Add categorical effects
                for col in ['department_name', 'prov_type', 'nlp_task', 
                           'model', 'category', 'subcategory', 'task_name']:
                    col_idx = f'{col}_idx'
                    if col_idx in encoded_cols:
                        effects_samples = categorical_samples[col][i, :]
                        indices = val_df[col_idx].values
                        valid_mask = indices >= 0
                        if valid_mask.any():
                            logit_p[valid_mask] += effects_samples[indices[valid_mask]]
                
                # Add TF-IDF effects
                if self.use_tfidf and n_tfidf_features > 0:
                    logit_p = logit_p + tfidf_matrix @ beta_tfidf_flat[i, :]
                
                predictions[i, :] = 1 / (1 + np.exp(-logit_p))
            
            mean_pred = predictions.mean(axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
            
            return {
                'mean': mean_pred,
                'lower_95': lower_ci,
                'upper_95': upper_ci,
                'full_samples': predictions
            }
        
        def compute_irt_parameters(self):
            """Compute IRT parameters with Bayesian uncertainty"""
            
            print("\n" + "="*80)
            print("COMPUTING BAYESIAN IRT PARAMETERS")
            print("="*80)
            
            epsilon = 1e-10
            
            # Compute parameters for all samples
            pred_prob_clipped = np.clip(self.df['pred_prob_mean'], epsilon, 1 - epsilon)
            self.df['difficulty_b'] = -np.log(pred_prob_clipped / (1 - pred_prob_clipped))
            
            lower_clipped = np.clip(self.df['pred_prob_lower'], epsilon, 1 - epsilon)
            upper_clipped = np.clip(self.df['pred_prob_upper'], epsilon, 1 - epsilon)
            self.df['difficulty_b_lower'] = -np.log(upper_clipped / (1 - upper_clipped))
            self.df['difficulty_b_upper'] = -np.log(lower_clipped / (1 - lower_clipped))
            
            uncertainty = self.df['pred_prob_upper'] - self.df['pred_prob_lower']
            self.df['discrimination_a_certainty'] = np.abs(self.df['pred_prob_mean'] - 0.5) * 2
            self.df['discrimination_a_uncertainty'] = 1 / (1 + uncertainty)
            self.df['discrimination_a'] = (
                self.df['discrimination_a_certainty'] *
                self.df['discrimination_a_uncertainty']
            )
            
            # Item type assignment
            self.df['item_type'] = 'Medium'
            self.df.loc[self.df['difficulty_b'] > self.df['difficulty_b'].quantile(0.75), 
                'item_type'] = 'Hard (likely rejected)'
            self.df.loc[self.df['difficulty_b'] < self.df['difficulty_b'].quantile(0.25), 
                'item_type'] = 'Easy (likely accepted)'
            
            # Performance metrics
            def compute_group_accuracy(df):
                results = {}

                easy = df[df['item_type'] == 'Easy (likely accepted)']
                if len(easy) > 0:
                    easy_acc = (easy['isPositive'].astype(int) == 1).mean()
                    results['easy_n'] = len(easy)
                    results['easy_acc'] = easy_acc
                else:
                    results['easy_n'] = 0
                    results['easy_acc'] = None

                hard = df[df['item_type'] == 'Hard (likely rejected)']
                if len(hard) > 0:
                    hard_acc = (hard['isPositive'].astype(int) == 0).mean()
                    results['hard_n'] = len(hard)
                    results['hard_acc'] = hard_acc
                else:
                    results['hard_n'] = 0
                    results['hard_acc'] = None
                
                return results
            
            overall_stats = compute_group_accuracy(self.df)
            
            # Print summary
            print(f"\nOVERALL (All folds combined):")
            print(f"  Mean difficulty (b): {self.df['difficulty_b'].mean():.3f}")
            print(f"  Mean discrimination (a): {self.df['discrimination_a'].mean():.3f}")
            print(f"  Easy items: n={overall_stats['easy_n']}, accuracy={overall_stats['easy_acc']:.3f}")
            print(f"  Hard items: n={overall_stats['hard_n']}, accuracy={overall_stats['hard_acc']:.3f}")

            # Overall AUC
            overall_auc = roc_auc_score(self.df['isPositive'].astype(int), self.df['pred_prob_mean'])
            print(f"  Overall AUC: {overall_auc:.4f}")
            
            # === CALIBRATION METRICS ===
            print("\n" + "="*80)
            print("CALIBRATION METRICS")
            print("="*80)
            
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            pred_probs = self.df['pred_prob_mean'].values
            true_labels = self.df['accepted'].values
            
            ece = 0.0
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = true_labels[in_bin].mean()
                    avg_confidence_in_bin = pred_probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_counts.append(in_bin.sum())
            
            print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
            print(f"(Lower is better; perfect calibration = 0.0)")
            
            # === CONFIDENCE-BASED ACCURACY ===
            print("\n" + "="*80)
            print("CONFIDENCE-BASED ACCURACY ANALYSIS")
            print("="*80)

            # Prediction confidence (distance from 0.5)
            self.df['confidence'] = np.abs(self.df['pred_prob_mean'] - 0.5)
            self.df['pred_binary'] = (self.df['pred_prob_mean'] > 0.5).astype(int)
            self.df['is_correct'] = (self.df['pred_binary'] == self.df['accepted']).astype(int)

            # Sort by confidence
            df_sorted = self.df.sort_values('confidence', ascending=False)

            # Top 25% most confident predictions
            top_25 = df_sorted.iloc[: int(len(df_sorted) * 0.25)]

            print(f"\nTop 25% Most Confident Predictions (n={len(top_25)}):")
            print(f"  Overall accuracy: {top_25['is_correct'].mean():.4f}")

            # Accuracy conditioned on predicted label
            top_pred1 = top_25[top_25['pred_binary'] == 1]
            top_pred0 = top_25[top_25['pred_binary'] == 0]

            acc_pred1 = top_pred1['is_correct'].mean() if len(top_pred1) > 0 else float('nan')
            acc_pred0 = top_pred0['is_correct'].mean() if len(top_pred0) > 0 else float('nan')

            print("\nAccuracy of confident predictions, conditioned on predicted label:")
            print(f"  When confidently predicting 1: accuracy = {acc_pred1:.4f} (n={len(top_pred1)})")
            print(f"  When confidently predicting 0: accuracy = {acc_pred0:.4f} (n={len(top_pred0)})")

            
            # Store calibration metrics
            self.ece = ece
            self.bin_accuracies = bin_accuracies
            self.bin_confidences = bin_confidences
            self.bin_counts = bin_counts

            return self
# Create combined text column
df['combined_text'] = df['user_query'].fillna('') + ' ' + df['chatehr_response'].fillna('')


bayes_model = BayesianIRT(
    df,
    n_folds=5,
    random_state=42,
    use_tfidf=True,
    tfidf_max_features=50,
    tfidf_ngram_range=(1, 2)
)

bayes_model.prepare_data()
bayes_model.fit_model(draws=1000, tune=500, chains=1, balance_classes=True)
bayes_model.compute_irt_parameters()
