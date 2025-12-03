import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
df = pd.read_csv("CS32h/sample_data.csv")

def compute_ece(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    ECE = Σ (n_k / n) * |acc_k - conf_k|
    where k iterates over bins, n_k is number of samples in bin k
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece

class NeuralIRT(nn.Module):
    """
    Neural Item Response Theory Model
    
    Feature-conditioned Rasch model:
    P(y_i = 1 | z_i, c_i) = σ(g_θ(c_i) - h_θ(z_i))
    
    where:
    - g_θ(c_i): context-dependent ability (task/model performance)
    - h_θ(z_i): embedding-derived difficulty (query complexity)
    """
    
    def __init__(self, query_dim, context_dim, hidden_dim=64):
        super(NeuralIRT, self).__init__()
        
        # Difficulty network: maps query features to difficulty
        self.difficulty_net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        
        # Ability network: maps context features to ability
        self.ability_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query_features, context_features):
        """
        Args:
            query_features: z_i (TF-IDF + prompt tokens)
            context_features: c_i (task type, model, department one-hot)
        
        Returns:
            logits: g_θ(c_i) - h_θ(z_i) (ability - difficulty)
            ability: g_θ(c_i)
            difficulty: h_θ(z_i)
        """
        difficulty = self.difficulty_net(query_features)  # h_θ(z_i)
        ability = self.ability_net(context_features)       # g_θ(c_i)
        
        logits = ability - difficulty  # Rasch model: ability - difficulty
        
        return logits, ability.squeeze(), difficulty.squeeze()
    
    def get_irt_parameters(self, query_features, context_features):
        """Extract interpretable IRT parameters"""
        with torch.no_grad():
            _, ability, difficulty = self.forward(query_features, context_features)
            
        return {
            'ability': ability.cpu().numpy(),      # θ in traditional IRT
            'difficulty': difficulty.cpu().numpy()  # b in traditional IRT
        }


class NeuralIRTModel:
    """
    Neural IRT Model Implementation with K-Fold Cross-Validation
    """
    
    def __init__(self, df, n_splits=5, random_state=42,
                 use_tfidf=True, tfidf_max_features=3072, 
                 tfidf_ngram_range=(1, 1)):
        self.df = df.copy()
        self.n_splits = n_splits
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TF-IDF parameters
        self.use_tfidf = use_tfidf
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        
        # Storage for cross-validation results
        self.fold_results = []
        self.fold_models = []
        
        print(f"Using device: {self.device}")
        print(f"K-Fold cross-validation with {n_splits} splits")
        
    def prepare_data(self):
        """Prepare query features (z_i) and context features (c_i)"""
        
        self.df['accepted'] = self.df['isPositive'].astype(int)
        
        # Create combined text for query embedding
        print("Creating query text features (z_i)...")
        self.df['combined_text'] = (
            self.df['user_query'].fillna('') + ' ' + 
            self.df['chatehr_response'].fillna('')
        )
        
        print("\n" + "="*80)
        print("NEURAL IRT - DATA PREPARATION")
        print("="*80)
        print(f"Total samples: {len(self.df)}")
        print(f"Acceptance rate: {self.df['accepted'].mean():.2%}")
        
        return self
    
    def _prepare_fold_features(self, train_idx, test_idx):
        """Prepare features for a specific fold"""

        train_df = self.df.iloc[train_idx].copy()
        test_df = self.df.iloc[test_idx].copy()

        # ============================================================
        # QUERY FEATURES (z_i): TF-IDF + numeric
        # ============================================================
        query_features_train = []
        query_features_test = []

        # TF-IDF embeddings
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

            train_text = train_df['combined_text'].fillna('')
            tfidf_train = tfidf_vectorizer.fit_transform(train_text).toarray()

            test_text = test_df['combined_text'].fillna('')
            tfidf_test = tfidf_vectorizer.transform(test_text).toarray()

            query_features_train.append(tfidf_train)
            query_features_test.append(tfidf_test)

        # Prompt tokens (scaled)
        if 'prompt_tokens' in train_df.columns:
            scaler = StandardScaler()
            median = train_df['prompt_tokens'].median()

            train_tokens = train_df[['prompt_tokens']].fillna(median)
            test_tokens = test_df[['prompt_tokens']].fillna(median)

            train_tokens_scaled = scaler.fit_transform(train_tokens)
            test_tokens_scaled = scaler.transform(test_tokens)

            query_features_train.append(train_tokens_scaled)
            query_features_test.append(test_tokens_scaled)

        query_features_train = np.hstack(query_features_train)
        query_features_test = np.hstack(query_features_test)

        # ============================================================
        # CONTEXT FEATURES (c_i): One-hot task descriptors
        # ============================================================
        context_features_train = []
        context_features_test = []

        encoders = {}
        feature_names = []

        categorical_features = ['prov_type', 'nlp_task', 'subcategory', 
                                'category', 'task_name', 'model', 'department_name']

        for col in categorical_features:
            if col in train_df.columns:
                unique_vals = train_df[col].fillna('missing').unique()
                n_unique = len(unique_vals)

                train_onehot = np.zeros((len(train_df), n_unique))
                test_onehot = np.zeros((len(test_df), n_unique))

                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}

                for i, val in enumerate(train_df[col].fillna('missing')):
                    if val in val_to_idx:
                        train_onehot[i, val_to_idx[val]] = 1

                for i, val in enumerate(test_df[col].fillna('missing')):
                    if val in val_to_idx:
                        test_onehot[i, val_to_idx[val]] = 1

                context_features_train.append(train_onehot)
                context_features_test.append(test_onehot)

                for val in unique_vals:
                    feature_names.append(f"{col}={val}")

                encoders[col] = val_to_idx

        context_features_train = np.hstack(context_features_train)
        context_features_test = np.hstack(context_features_test)

        # ============================================================
        # === NEW FOR CLASS BALANCING =================================
        # ============================================================
        labels = train_df['accepted'].values
        classes, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        class_weights = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
        sample_weights_train = np.array([class_weights[y] for y in labels])

        return {
            'train_df': train_df,
            'test_df': test_df,
            'query_features_train': query_features_train,
            'query_features_test': query_features_test,
            'context_features_train': context_features_train,
            'context_features_test': context_features_test,
            'encoders': encoders,
            'feature_names': feature_names,

            ### ← NEW RETURNS
            'sample_weights_train': sample_weights_train,
            'class_weights': class_weights
        }

    
    def fit(self, hidden_dim=64, epochs=50, batch_size=64, 
            lr=0.001, weight_decay=1e-4, class_weight=None):

        print("\n" + "="*80)
        print("FITTING NEURAL IRT MODEL WITH K-FOLD CROSS-VALIDATION")
        print("="*80)

        # Tracking lists (FIXED: these were missing!)
        all_train_aucs, all_test_aucs = [], []
        all_train_eces, all_test_eces = [], []
        all_train_acc_top25, all_test_acc_top25 = [], []
        all_train_acc_conf_true, all_test_acc_conf_true = [], []

        skf = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.df, self.df['accepted'])):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*80}")

            fold_data = self._prepare_fold_features(train_idx, test_idx)

            train_df = fold_data['train_df']
            test_df = fold_data['test_df']
            query_features_train = fold_data['query_features_train']
            query_features_test = fold_data['query_features_test']
            context_features_train = fold_data['context_features_train']
            context_features_test = fold_data['context_features_test']
            sample_weights_train = fold_data['sample_weights_train']
            class_weights = fold_data['class_weights']

            class_weight_tensor = torch.tensor(
                [class_weights[c] for c in sorted(class_weights.keys())],
                dtype=torch.float32
            ).to(self.device)

            query_dim = query_features_train.shape[1]
            context_dim = context_features_train.shape[1]

            model = NeuralIRT(query_dim=query_dim,
                            context_dim=context_dim,
                            hidden_dim=hidden_dim).to(self.device)

            train_dataset = TensorDataset(
                torch.FloatTensor(query_features_train),
                torch.FloatTensor(context_features_train),
                torch.FloatTensor(train_df['accepted'].values),
                torch.FloatTensor(sample_weights_train)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1])
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            print(f"\nTraining fold {fold_idx + 1}...")
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                correct = 0
                total = 0

                for query_batch, context_batch, labels_batch, sw_batch in train_loader:
                    query_batch = query_batch.to(self.device)
                    context_batch = context_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    sw_batch = sw_batch.to(self.device)

                    logits, _, _ = model(query_batch, context_batch)

                    loss = criterion(logits.squeeze(), labels_batch)
                    loss = (loss * sw_batch).mean()  # sample-weighted

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(labels_batch)
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                    correct += (preds == labels_batch).sum().item()
                    total += len(labels_batch)

                epoch_loss /= total
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

            fold_result = self._evaluate_fold(
                model,
                train_df, test_df,
                query_features_train, query_features_test,
                context_features_train, context_features_test,
                fold_idx
            )

            self.fold_results.append(fold_result)
            self.fold_models.append(model)

            all_train_aucs.append(fold_result['train_auc'])
            all_test_aucs.append(fold_result['test_auc'])
            all_train_eces.append(fold_result['train_ece'])
            all_test_eces.append(fold_result['test_ece'])
            all_train_acc_top25.append(fold_result['train_acc_top25_all'])
            all_test_acc_top25.append(fold_result['test_acc_top25_all'])
            all_train_acc_conf_true.append(fold_result['train_acc_conf_true'])
            all_test_acc_conf_true.append(fold_result['test_acc_conf_true'])

        # Summary
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)

        print(f"\nTRAIN (avg {self.n_splits} folds): AUC={np.mean(all_train_aucs):.4f} ± {np.std(all_train_aucs):.4f}")
        print(f"TEST                AUC={np.mean(all_test_aucs):.4f} ± {np.std(all_test_aucs):.4f}")

        self.cv_metrics = {
            'train_auc_mean': np.mean(all_train_aucs),
            'test_auc_mean': np.mean(all_test_aucs),
            'train_ece_mean': np.mean(all_train_eces),
            'test_ece_mean': np.mean(all_test_eces),
            'train_acc_top25_mean': np.mean(all_train_acc_top25),
            'test_acc_top25_mean': np.mean(all_test_acc_top25),
            'train_acc_conf_true_mean': np.mean(all_train_acc_conf_true),
            'test_acc_conf_true_mean': np.mean(all_test_acc_conf_true),
            'fold_results': self.fold_results
        }

        print("\n✓ Cross-validation complete!")
        return self

    
    def _evaluate_fold(self, model, train_df, test_df,
                      query_features_train, query_features_test,
                      context_features_train, context_features_test,
                     fold_idx):
        """Evaluate a single fold with comprehensive IRT metrics"""
        
        # Train predictions
        train_results = self._predict(
            model,
            query_features_train,
            context_features_train
        )
        
        train_df = train_df.copy()
        train_df['pred_prob'] = train_results['probabilities']
        train_df['ability_theta'] = train_results['ability']
        train_df['difficulty_b'] = train_results['difficulty']
        
        # Test predictions
        test_results = self._predict(
            model,
            query_features_test,
            context_features_test
        )
        
        test_df = test_df.copy()
        test_df['pred_prob'] = test_results['probabilities']
        test_df['ability_theta'] = test_results['ability']
        test_df['difficulty_b'] = test_results['difficulty']
        
        # Compute discrimination (how well ability - difficulty predicts)
        # Discrimination = gradient of sigmoid at ability - difficulty
        for df in [train_df, test_df]:
            logits = df['ability_theta'] - df['difficulty_b']
            probs = 1 / (1 + np.exp(-logits))
            # Discrimination = derivative of sigmoid = p(1-p)
            df['discrimination_a'] = probs * (1 - probs) * 2.5  # Scale factor
        
        # Compute confidence (distance from 0.5)
        train_df['confidence'] = np.abs(train_df['pred_prob'] - 0.5)
        test_df['confidence'] = np.abs(test_df['pred_prob'] - 0.5)
        
        # Item type assignment
        for df in [train_df, test_df]:
            df['item_type'] = 'Medium'
            df.loc[df['difficulty_b'] > df['difficulty_b'].quantile(0.75), 
                   'item_type'] = 'Hard (likely rejected)'
            df.loc[df['difficulty_b'] < df['difficulty_b'].quantile(0.25), 
                   'item_type'] = 'Easy (likely accepted)'
        
        # Performance metrics
        train_auc = roc_auc_score(train_df['accepted'], train_df['pred_prob'])
        test_auc = roc_auc_score(test_df['accepted'], test_df['pred_prob'])
        
        # Compute ECE
        train_ece = self._compute_ece(
            train_df['accepted'].values, 
            train_df['pred_prob'].values, 
            n_bins=10
        )
        test_ece = self._compute_ece(
            test_df['accepted'].values, 
            test_df['pred_prob'].values, 
            n_bins=10
        )
        
        # Confidence-based accuracy
        # Top 25% most confident predictions
        train_top25_idx = train_df.nlargest(int(len(train_df) * 0.25), 'confidence').index
        test_top25_idx = test_df.nlargest(int(len(test_df) * 0.25), 'confidence').index
        
        train_pred_binary = (train_df['pred_prob'] > 0.5).astype(int)
        test_pred_binary = (test_df['pred_prob'] > 0.5).astype(int)
        
        train_acc_top25_all = accuracy_score(
            train_df.loc[train_top25_idx, 'accepted'],
            train_pred_binary.loc[train_top25_idx]
        )
        test_acc_top25_all = accuracy_score(
            test_df.loc[test_top25_idx, 'accepted'],
            test_pred_binary.loc[test_top25_idx]
        )
        
        # Top 25% most confident TRUE predictions (pred_prob > 0.5)
        train_confident_true = train_df[train_df['pred_prob'] > 0.5].nlargest(
            int(len(train_df[train_df['pred_prob'] > 0.5]) * 0.25), 'pred_prob'
        )
        test_confident_true = test_df[test_df['pred_prob'] > 0.5].nlargest(
            int(len(test_df[test_df['pred_prob'] > 0.5]) * 0.25), 'pred_prob'
        )
        
        train_acc_conf_true = accuracy_score(
            train_confident_true['accepted'],
            [1] * len(train_confident_true)
        ) if len(train_confident_true) > 0 else 0
        
        test_acc_conf_true = accuracy_score(
            test_confident_true['accepted'],
            [1] * len(test_confident_true)
        ) if len(test_confident_true) > 0 else 0
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  TRAIN - AUC: {train_auc:.4f}, ECE: {train_ece:.4f}, Acc(top25%): {train_acc_top25_all:.4f}, Acc(conf TRUE): {train_acc_conf_true:.4f}")
        print(f"  TEST  - AUC: {test_auc:.4f}, ECE: {test_ece:.4f}, Acc(top25%): {test_acc_top25_all:.4f}, Acc(conf TRUE): {test_acc_conf_true:.4f}")
        print(f"  Mean ability (θ): {test_df['ability_theta'].mean():.3f} ± {test_df['ability_theta'].std():.3f}")
        print(f"  Mean difficulty (b): {test_df['difficulty_b'].mean():.3f} ± {test_df['difficulty_b'].std():.3f}")
        print(f"  Mean discrimination (a): {test_df['discrimination_a'].mean():.3f}")
        
        return {
            'fold_idx': fold_idx,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_ece': train_ece,
            'test_ece': test_ece,
            'train_acc_top25_all': train_acc_top25_all,
            'test_acc_top25_all': test_acc_top25_all,
            'train_acc_conf_true': train_acc_conf_true,
            'test_acc_conf_true': test_acc_conf_true,
            'train_df': train_df,
            'test_df': test_df,
            # 'feature_names': feature_names,
            'n_confident_true_train': len(train_confident_true),
            'n_confident_true_test': len(test_confident_true)
        }
    
    def _predict(self, model, query_features, context_features):
        """Make predictions with a specific model"""
        model.eval()
        
        query_tensor = torch.FloatTensor(query_features).to(self.device)
        context_tensor = torch.FloatTensor(context_features).to(self.device)
        
        with torch.no_grad():
            logits, ability, difficulty = model(query_tensor, context_tensor)
            probs = torch.sigmoid(logits.squeeze())
        
        return {
            'probabilities': probs.cpu().numpy(),
            'ability': ability.cpu().numpy(),
            'difficulty': difficulty.cpu().numpy(),
            'logits': logits.squeeze().cpu().numpy()
        }
    
    def _compute_ece(self, y_true, y_pred, n_bins=10):
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate(self):
        """
        Comprehensive evaluation after K-Fold CV
        Provides detailed analysis of IRT parameters and model performance
        """
        
        if not hasattr(self, 'fold_results'):
            raise ValueError("Must call fit() before evaluate()")
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Aggregate all test predictions
        all_test_dfs = [fold['test_df'] for fold in self.fold_results]
        combined_test = pd.concat(all_test_dfs, ignore_index=False).sort_index()
        
        # Aggregate all train predictions
        all_train_dfs = [fold['train_df'] for fold in self.fold_results]
        combined_train = pd.concat(all_train_dfs, ignore_index=False).sort_index()
        
        # Store for later access
        self.train_df = combined_train
        self.test_df = combined_test
        
        print(f"\n{'='*80}")
        print("OVERALL PERFORMANCE METRICS")
        print(f"{'='*80}")
        
        # Overall AUC and ECE on combined predictions
        overall_test_auc = roc_auc_score(combined_test['accepted'], combined_test['pred_prob'])
        overall_test_ece = self._compute_ece(
            combined_test['accepted'].values, 
            combined_test['pred_prob'].values, 
            n_bins=10
        )
        
        overall_train_auc = roc_auc_score(combined_train['accepted'], combined_train['pred_prob'])
        overall_train_ece = self._compute_ece(
            combined_train['accepted'].values, 
            combined_train['pred_prob'].values, 
            n_bins=10
        )
        
        print(f"\nAGGREGATED TEST SET (all folds combined):")
        print(f"  Total samples: {len(combined_test)}")
        print(f"  AUC: {overall_test_auc:.4f}")
        print(f"  ECE: {overall_test_ece:.4f}")
        print(f"  Acceptance rate: {combined_test['accepted'].mean():.2%}")
        
        print(f"\nAGGREGATED TRAIN SET (all folds combined):")
        print(f"  Total samples: {len(combined_train)}")
        print(f"  AUC: {overall_train_auc:.4f}")
        print(f"  ECE: {overall_train_ece:.4f}")
        
        # IRT Parameters Analysis
        print(f"\n{'='*80}")
        print("IRT PARAMETERS ANALYSIS (Test Set)")
        print(f"{'='*80}")
        
        print(f"\nABILITY (θ) - Task/Context Quality:")
        print(f"  Mean: {combined_test['ability_theta'].mean():.3f}")
        print(f"  Std: {combined_test['ability_theta'].std():.3f}")
        print(f"  Min: {combined_test['ability_theta'].min():.3f}")
        print(f"  Max: {combined_test['ability_theta'].max():.3f}")
        print(f"  Median: {combined_test['ability_theta'].median():.3f}")
        
        print(f"\nDIFFICULTY (b) - Query Difficulty:")
        print(f"  Mean: {combined_test['difficulty_b'].mean():.3f}")
        print(f"  Std: {combined_test['difficulty_b'].std():.3f}")
        print(f"  Min: {combined_test['difficulty_b'].min():.3f}")
        print(f"  Max: {combined_test['difficulty_b'].max():.3f}")
        print(f"  Median: {combined_test['difficulty_b'].median():.3f}")
        
        print(f"\nDISCRIMINATION (a) - Prediction Certainty:")
        print(f"  Mean: {combined_test['discrimination_a'].mean():.3f}")
        print(f"  Std: {combined_test['discrimination_a'].std():.3f}")
        print(f"  Min: {combined_test['discrimination_a'].min():.3f}")
        print(f"  Max: {combined_test['discrimination_a'].max():.3f}")
        
        # Item difficulty distribution
        print(f"\n{'='*80}")
        print("ITEM DIFFICULTY DISTRIBUTION")
        print(f"{'='*80}")
        
        item_dist = combined_test['item_type'].value_counts()
        for item_type in ['Easy (likely accepted)', 'Medium', 'Hard (likely rejected)']:
            if item_type in item_dist.index:
                count = item_dist[item_type]
                pct = count / len(combined_test) * 100
                print(f"  {item_type}: {count} samples ({pct:.1f}%)")
        
        # Accuracy by item type
        print(f"\nAccuracy by Item Difficulty:")
        for item_type in combined_test['item_type'].unique():
            mask = combined_test['item_type'] == item_type
            subset = combined_test[mask]
            preds = (subset['pred_prob'] > 0.5).astype(int)
            acc = accuracy_score(subset['accepted'], preds)
            print(f"  {item_type}: {acc:.4f} (n={len(subset)})")
        
        # Confidence analysis
        print(f"\n{'='*80}")
        print("CONFIDENCE-BASED ACCURACY")
        print(f"{'='*80}")
        
        pct = 25
        n_samples = int(len(combined_test) * pct / 100)
        top_confident = combined_test.nlargest(n_samples, 'confidence')

        # Predicted labels
        preds = (top_confident['pred_prob'] > 0.5).astype(int)
        true = top_confident['accepted']

        # Split by true label
        accepted_mask = (true == 1)
        rejected_mask = (true == 0)

        acc_accept = accuracy_score(true[accepted_mask], preds[accepted_mask]) if accepted_mask.any() else float('nan')
        acc_reject = accuracy_score(true[rejected_mask], preds[rejected_mask]) if rejected_mask.any() else float('nan')

        print(f"Top {pct}% most confident:")
        print(f"  Accuracy on ACCEPTED class: {acc_accept:.4f} (n={accepted_mask.sum()})")
        print(f"  Accuracy on REJECTED class: {acc_reject:.4f} (n={rejected_mask.sum()})")        # Top quartiles
        for pct in [10, 25, 50]:
            n_samples = int(len(combined_test) * pct / 100)
            top_confident = combined_test.nlargest(n_samples, 'confidence')
            preds = (top_confident['pred_prob'] > 0.5).astype(int)
            acc = accuracy_score(top_confident['accepted'], preds)
            avg_conf = top_confident['confidence'].mean()
            print(f"  Top {pct}% most confident: Accuracy = {acc:.4f}, Avg confidence = {avg_conf:.4f}")
        
        # Confident acceptances
        print(f"\nConfident Acceptances (pred_prob > 0.5):")
        confident_accepts = combined_test[combined_test['pred_prob'] > 0.5]
        for pct in [10, 25, 50]:
            n_samples = int(len(confident_accepts) * pct / 100)
            if n_samples > 0:
                top = confident_accepts.nlargest(n_samples, 'pred_prob')
                acc = accuracy_score(top['accepted'], [1] * len(top))
                avg_prob = top['pred_prob'].mean()
                print(f"  Top {pct}%: Accuracy = {acc:.4f}, Avg prob = {avg_prob:.4f} (n={len(top)})")
        
        # Confident rejections
        print(f"\nConfident Rejections (pred_prob < 0.5):")
        confident_rejects = combined_test[combined_test['pred_prob'] < 0.5]
        for pct in [10, 25, 50]:
            n_samples = int(len(confident_rejects) * pct / 100)
            if n_samples > 0:
                bottom = confident_rejects.nsmallest(n_samples, 'pred_prob')
                acc = accuracy_score(bottom['accepted'], [0] * len(bottom))
                avg_prob = bottom['pred_prob'].mean()
                print(f"  Top {pct}%: Accuracy = {acc:.4f}, Avg prob = {avg_prob:.4f} (n={len(bottom)})")
        
        # Calibration by bins
        print(f"\n{'='*80}")
        print("CALIBRATION ANALYSIS")
        print(f"{'='*80}")
        
        bins = np.linspace(0, 1, 11)
        for i in range(len(bins) - 1):
            bin_lower, bin_upper = bins[i], bins[i+1]
            mask = (combined_test['pred_prob'] >= bin_lower) & (combined_test['pred_prob'] < bin_upper)
            if mask.sum() > 0:
                subset = combined_test[mask]
                actual_rate = subset['accepted'].mean()
                predicted_rate = subset['pred_prob'].mean()
                diff = predicted_rate - actual_rate
                print(f"  [{bin_lower:.1f}, {bin_upper:.1f}): n={mask.sum():4d}, "
                      f"Pred={predicted_rate:.3f}, Actual={actual_rate:.3f}, "
                      f"Diff={diff:+.3f}")
        
        # Cross-validation stability
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION STABILITY")
        print(f"{'='*80}")
        
        test_aucs = [fold['test_auc'] for fold in self.fold_results]
        test_eces = [fold['test_ece'] for fold in self.fold_results]
        
        print(f"\nTest AUC across folds:")
        print(f"  Mean: {np.mean(test_aucs):.4f}")
        print(f"  Std: {np.std(test_aucs):.4f}")
        print(f"  Min: {np.min(test_aucs):.4f}")
        print(f"  Max: {np.max(test_aucs):.4f}")
        print(f"  Range: {np.max(test_aucs) - np.min(test_aucs):.4f}")
        
        print(f"\nTest ECE across folds:")
        print(f"  Mean: {np.mean(test_eces):.4f}")
        print(f"  Std: {np.std(test_eces):.4f}")
        print(f"  Min: {np.min(test_eces):.4f}")
        print(f"  Max: {np.max(test_eces):.4f}")
        
        # Store comprehensive metrics
        self.test_auc = overall_test_auc
        self.test_ece = overall_test_ece
        self.metrics = {
            'overall': {
                'train_auc': overall_train_auc,
                'train_ece': overall_train_ece,
                'test_auc': overall_test_auc,
                'test_ece': overall_test_ece
            },
            'cv': self.cv_metrics
        }
        
        print("\n✓ Evaluation complete!")
        
        return self


    
    def get_aggregated_predictions(self):
        """
        Get predictions aggregated across all folds
        Each sample appears in exactly one test fold
        """
        all_predictions = []
        
        for fold_result in self.fold_results:
            test_df = fold_result['test_df'].copy()
            test_df['fold'] = fold_result['fold_idx']
            all_predictions.append(test_df)
        
        aggregated_df = pd.concat(all_predictions, ignore_index=False).sort_index()
        
        return aggregated_df



# Initialize model
model = NeuralIRTModel(
    df,
    random_state=42,
    use_tfidf=True,
    tfidf_max_features=3072,
    tfidf_ngram_range=(1, 2)
)

# Prepare data
model.prepare_data()

# Fit model
model.fit(
    hidden_dim=64,
    epochs=50,
    batch_size=64,
    lr=0.001,
    weight_decay=1e-4
)

# Evaluate (includes ECE and confidence metrics)
model.evaluate()

# Access results
print("\n" + "="*80)
print("TOP 10 HARDEST QUERIES (highest difficulty)")
print("="*80)
hardest = model.test_df.nlargest(10, 'difficulty_b')
print(hardest[['user_query', 'difficulty_b', 'ability_theta', 'accepted']])

print("\n" + "="*80)
print("TOP 10 EASIEST QUERIES (lowest difficulty)")
print("="*80)
easiest = model.test_df.nsmallest(10, 'difficulty_b')
print(easiest[['user_query', 'difficulty_b', 'ability_theta', 'accepted']])
