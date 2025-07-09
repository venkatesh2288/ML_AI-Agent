import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

class ContextTreeNode:
    """
    Represents a node in the context tree.
    Each node contains feature information and relationships.
    """
    def __init__(self, feature_name, importance_score=0.0, context_level=0):
        self.feature_name = feature_name
        self.importance_score = importance_score
        self.context_level = context_level
        self.children = []
        self.parent = None
        self.related_features = []
        self.statistical_properties = {}
        
    def add_child(self, child_node):
        """Add a child node to this node."""
        child_node.parent = self
        child_node.context_level = self.context_level + 1
        self.children.append(child_node)
    
    def add_related_feature(self, feature_name, relationship_strength):
        """Add a related feature with its relationship strength."""
        self.related_features.append({
            'feature': feature_name,
            'strength': relationship_strength
        })

class ContextTreeFeatureSelector:
    """
    Context Tree Feature Selector that builds a hierarchical tree of feature relationships
    and selects features based on contextual importance and statistical properties.
    
    Args:
        X (pd.DataFrame): The training feature data.
        y (pd.Series): The training target data.
        problem_type (str): 'classification' or 'regression'
        max_features_ratio (float): Maximum ratio of features to select (0.3 = 30% of features)
        min_importance_threshold (float): Minimum importance threshold for feature selection
        correlation_threshold (float): Correlation threshold for grouping related features
        random_state (int): Random state for reproducibility
    """
    
    def __init__(self, X, y, problem_type='classification', max_features_ratio=0.7, 
                 min_importance_threshold=0.05, correlation_threshold=0.7, random_state=42):
        self.X = X
        self.y = y
        self.problem_type = problem_type.lower()
        self.max_features_ratio = max_features_ratio
        self.min_importance_threshold = min_importance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        
        self.feature_names = X.columns.tolist()
        self.n_features = len(self.feature_names)
        self.max_features = max(1, int(self.n_features * max_features_ratio))
        
        # Tree structure
        self.root = None
        self.nodes = {}
        self.selected_features = []
        
        # Feature analysis results
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        self.mutual_info_scores = {}
        
    def _calculate_feature_importance(self):
        """Calculate feature importance using multiple methods."""
        print("  - Calculating feature importance scores...")
        
        # Method 1: Random Forest Feature Importance
        if self.problem_type == 'classification':
            rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        # Handle categorical features for mutual information
        X_encoded = self.X.copy()
        categorical_features = []
        
        for i, col in enumerate(X_encoded.columns):
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                categorical_features.append(i)
        
        try:
            rf_model.fit(X_encoded, self.y)
            rf_importance = rf_model.feature_importances_
        except:
            rf_importance = np.ones(self.n_features) / self.n_features
        
        # Method 2: Mutual Information
        try:
            if self.problem_type == 'classification':
                mi_scores = mutual_info_classif(X_encoded, self.y, discrete_features=categorical_features, random_state=self.random_state)
            else:
                mi_scores = mutual_info_regression(X_encoded, self.y, discrete_features=categorical_features, random_state=self.random_state)
            
            # Normalize MI scores
            if mi_scores.max() > 0:
                mi_scores = mi_scores / mi_scores.max()
        except:
            mi_scores = np.ones(self.n_features) / self.n_features
        
        # Method 3: Simple correlation with target (for numerical features)
        target_correlation = np.zeros(self.n_features)
        for i, col in enumerate(self.feature_names):
            if pd.api.types.is_numeric_dtype(self.X[col]) and pd.api.types.is_numeric_dtype(self.y):
                try:
                    corr = abs(np.corrcoef(self.X[col], self.y)[0, 1])
                    target_correlation[i] = corr if not np.isnan(corr) else 0
                except:
                    target_correlation[i] = 0
        
        # Combine importance scores (weighted average)
        combined_importance = (0.4 * rf_importance + 0.4 * mi_scores + 0.2 * target_correlation)
        
        # Store individual feature importance
        for i, feature in enumerate(self.feature_names):
            self.feature_importance_scores[feature] = {
                'combined': combined_importance[i],
                'rf_importance': rf_importance[i],
                'mutual_info': mi_scores[i],
                'target_correlation': target_correlation[i]
            }
    
    def _calculate_feature_correlations(self):
        """Calculate correlation matrix for numerical features."""
        print("  - Calculating feature correlations...")
        
        # Select only numerical columns for correlation
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            self.correlation_matrix = self.X[numerical_cols].corr().abs()
        else:
            # If no numerical columns, create identity matrix
            self.correlation_matrix = pd.DataFrame(np.eye(len(self.feature_names)), 
                                                 index=self.feature_names, 
                                                 columns=self.feature_names)
    
    def _build_context_tree(self):
        """Build the context tree based on feature relationships."""
        print("  - Building context tree structure...")
        
        # Create root node
        self.root = ContextTreeNode("ROOT", importance_score=1.0, context_level=0)
        
        # Sort features by combined importance score
        sorted_features = sorted(self.feature_names, 
                               key=lambda x: self.feature_importance_scores[x]['combined'], 
                               reverse=True)
        
        # Create nodes for each feature
        for feature in sorted_features:
            importance = self.feature_importance_scores[feature]['combined']
            node = ContextTreeNode(feature, importance_score=importance, context_level=1)
            self.nodes[feature] = node
            
            # Add statistical properties
            node.statistical_properties = {
                'rf_importance': self.feature_importance_scores[feature]['rf_importance'],
                'mutual_info': self.feature_importance_scores[feature]['mutual_info'],
                'target_correlation': self.feature_importance_scores[feature]['target_correlation'],
                'data_type': str(self.X[feature].dtype),
                'unique_values': self.X[feature].nunique(),
                'missing_ratio': self.X[feature].isnull().sum() / len(self.X)
            }
        
        # Build hierarchical relationships based on correlation and importance
        self._build_hierarchical_relationships()
    
    def _build_hierarchical_relationships(self):
        """Build hierarchical relationships between features."""
        # Start with the most important features as primary nodes
        primary_features = sorted(self.feature_names, 
                                key=lambda x: self.feature_importance_scores[x]['combined'], 
                                reverse=True)[:max(1, self.n_features // 3)]
        
        # Add primary features directly to root
        for feature in primary_features:
            if feature in self.nodes:
                self.root.add_child(self.nodes[feature])
        
        # Group remaining features based on correlation with primary features
        remaining_features = [f for f in self.feature_names if f not in primary_features]
        
        for feature in remaining_features:
            if feature not in self.nodes:
                continue
                
            best_parent = self.root
            best_relationship_strength = 0
            
            # Find the best parent based on correlation
            for primary_feature in primary_features:
                if (feature in self.correlation_matrix.columns and 
                    primary_feature in self.correlation_matrix.columns):
                    
                    correlation = self.correlation_matrix.loc[feature, primary_feature]
                    if correlation > best_relationship_strength and correlation > 0.3:
                        best_relationship_strength = correlation
                        best_parent = self.nodes[primary_feature]
            
            # Add as child to best parent
            if best_parent != self.root:
                best_parent.add_child(self.nodes[feature])
                self.nodes[feature].add_related_feature(best_parent.feature_name, best_relationship_strength)
            else:
                self.root.add_child(self.nodes[feature])
    
    def _select_features_from_tree(self):
        """Select features from the context tree based on various criteria."""
        print("  - Selecting features from context tree...")
        
        feature_scores = []
        
        for feature in self.feature_names:
            if feature not in self.nodes:
                continue
                
            node = self.nodes[feature]
            
            # Calculate contextual score
            contextual_score = node.importance_score
            
            # Bonus for being a primary node (direct child of root)
            if node.parent == self.root:
                contextual_score *= 1.2
            
            # Bonus for having strong relationships
            if node.related_features:
                avg_relationship_strength = np.mean([rel['strength'] for rel in node.related_features])
                contextual_score *= (1 + 0.1 * avg_relationship_strength)
            
            # Penalty for high correlation with already selected features
            correlation_penalty = 0
            if hasattr(self, 'temp_selected') and self.temp_selected:
                for selected_feature in self.temp_selected:
                    if (feature in self.correlation_matrix.columns and 
                        selected_feature in self.correlation_matrix.columns):
                        correlation = self.correlation_matrix.loc[feature, selected_feature]
                        if correlation > self.correlation_threshold:
                            correlation_penalty += 0.2
            
            final_score = max(0, contextual_score - correlation_penalty)
            
            feature_scores.append({
                'feature': feature,
                'score': final_score,
                'node': node
            })
        
        # Sort by score and select top features
        feature_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select features ensuring diversity and respecting thresholds
        self.selected_features = []
        self.temp_selected = []
        
        for item in feature_scores:
            if len(self.selected_features) >= self.max_features:
                break
                
            feature = item['feature']
            score = item['score']
            
            # Check minimum importance threshold
            if score < self.min_importance_threshold:
                continue
            
            # Check correlation with already selected features
            highly_correlated = False
            for selected_feature in self.selected_features:
                if (feature in self.correlation_matrix.columns and 
                    selected_feature in self.correlation_matrix.columns):
                    correlation = self.correlation_matrix.loc[feature, selected_feature]
                    if correlation > self.correlation_threshold:
                        highly_correlated = True
                        break
            
            if not highly_correlated:
                self.selected_features.append(feature)
                self.temp_selected.append(feature)
        
        # Ensure minimum number of features
        if len(self.selected_features) < 2:
            # Add top features regardless of correlation
            for item in feature_scores:
                if len(self.selected_features) >= max(2, self.max_features):
                    break
                if item['feature'] not in self.selected_features:
                    self.selected_features.append(item['feature'])
        
        delattr(self, 'temp_selected')
    
    def run(self):
        """Run the complete context tree feature selection process."""
        print("\n--- Running Context Tree Feature Selection ---")
        
        # Step 1: Calculate feature importance
        self._calculate_feature_importance()
        
        # Step 2: Calculate feature correlations
        self._calculate_feature_correlations()
        
        # Step 3: Build context tree
        self._build_context_tree()
        
        # Step 4: Select features from tree
        self._select_features_from_tree()
        
        print(f"  - Context Tree selected {len(self.selected_features)} features from {self.n_features} total features")
        print(f"  - Selected features: {self.selected_features}")
        
        # Return selected features and their importance scores
        selected_feature_info = {}
        for feature in self.selected_features:
            selected_feature_info[feature] = self.feature_importance_scores[feature]['combined']
        
        return self.selected_features, selected_feature_info
    
    def get_tree_summary(self):
        """Get a summary of the context tree structure."""
        if not self.root:
            return "Context tree not built yet."
        
        summary = []
        summary.append(f"Context Tree Summary:")
        summary.append(f"Total features: {self.n_features}")
        summary.append(f"Selected features: {len(self.selected_features)}")
        summary.append(f"Selection ratio: {len(self.selected_features)/self.n_features:.2%}")
        summary.append(f"Primary nodes (Level 1): {len(self.root.children)}")
        
        # Count secondary nodes
        secondary_count = 0
        for child in self.root.children:
            secondary_count += len(child.children)
        summary.append(f"Secondary nodes (Level 2): {secondary_count}")
        
        return "\n".join(summary)