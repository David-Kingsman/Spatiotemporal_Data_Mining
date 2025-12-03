"""Tree-based Baselines: RF, XGBoost, LightGBM"""
import numpy as np
from typing import Optional, Literal, Tuple
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class TreeConfig:
    """Tree Model Configuration"""
    model_type: Literal["rf", "xgb", "lgbm"] = "xgb"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    quantile_regression: bool = False  # Whether to use quantile regression
    quantiles: list = None  # List of quantiles, e.g., [0.1, 0.5, 0.9]


class TreeBaseline:
    """Tree-based Baseline Model"""
    
    def __init__(self, config: TreeConfig):
        self.config = config
        self.model = None
        self.quantile_models = {}  # For quantile regression
        
        if config.quantiles is None:
            config.quantiles = [0.1, 0.5, 0.9]
        
        # Fallback if specified model is not available
        if config.model_type == "xgb" and not XGBOOST_AVAILABLE:
            print("Warning: XGBoost not available, falling back to Random Forest")
            config.model_type = "rf"
        elif config.model_type == "lgbm" and not LIGHTGBM_AVAILABLE:
            print("Warning: LightGBM not available, falling back to Random Forest")
            config.model_type = "rf"
    
    def _create_model(self, quantile: Optional[float] = None):
        """Create model instance"""
        if self.config.model_type == "rf":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required")
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.config.model_type == "xgb":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost is required")
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth or 6,
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
            if quantile is not None:
                params['objective'] = f'reg:quantileerror'
                params['quantile_alpha'] = quantile
            return xgb.XGBRegressor(**params)
        
        elif self.config.model_type == "lgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("lightgbm is required")
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth or -1,
                'random_state': 42,
                'objective': 'regression'
            }
            if quantile is not None:
                params['objective'] = 'quantile'
                params['alpha'] = quantile
            return lgb.LGBMRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train model"""
        if self.config.quantile_regression:
            # Train a model for each quantile
            for q in self.config.quantiles:
                model = self._create_model(quantile=q)
                model.fit(X, y)
                self.quantile_models[q] = model
        else:
            # Standard regression
            self.model = self._create_model()
            self.model.fit(X, y)
            # For RF, calculate training error for uncertainty estimation
            if self.config.model_type == "rf":
                y_pred_train = self.model.predict(X)
                self.train_std = np.std(y - y_pred_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean"""
        if self.config.quantile_regression:
            # Use median as prediction
            if 0.5 in self.quantile_models:
                return self.quantile_models[0.5].predict(X)
            else:
                # If no 0.5, use mean of all quantiles
                predictions = [model.predict(X) for model in self.quantile_models.values()]
                return np.mean(predictions, axis=0)
        else:
            return self.model.predict(X)
    
    def predict_quantiles(self, X: np.ndarray) -> dict:
        """Predict quantiles"""
        if not self.config.quantile_regression:
            raise ValueError("quantile_regression=True is required")
        
        results = {}
        for q, model in self.quantile_models.items():
            results[q] = model.predict(X)
        return results
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation (estimated from quantiles)
        
        Returns:
        mean: Predicted mean
        std: Estimated standard deviation
        """
        if self.config.quantile_regression:
            quantiles = self.predict_quantiles(X)
            # Estimate mean
            if 0.5 in quantiles:
                mean = quantiles[0.5]
            else:
                mean = np.mean([q for q in quantiles.values()], axis=0)
            
            # Estimate std from quantiles (assuming normal distribution)
            if 0.1 in quantiles and 0.9 in quantiles:
                # Estimate std using 10% and 90% quantiles
                std = (quantiles[0.9] - quantiles[0.1]) / (2 * 1.28)  # 1.28 is z-score for 90% interval
            elif 0.25 in quantiles and 0.75 in quantiles:
                std = (quantiles[0.75] - quantiles[0.25]) / (2 * 0.675)  # IQR
            else:
                # Estimate using all quantiles
                all_quantiles = np.array(list(quantiles.values()))
                std = np.std(all_quantiles, axis=0)
            
            return mean, std
        else:
            # For non-quantile models, use empirical estimation
            mean = self.predict(X)
            
            # For RF, use std of all trees
            if self.config.model_type == "rf" and hasattr(self.model, 'estimators_'):
                # Get predictions from all trees
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                std = np.std(tree_predictions, axis=0)
            # Use training error std
            elif hasattr(self, 'train_std'):
                std = np.full(len(mean), self.train_std)
            else:
                # Heuristic: use a percentage of predicted value
                std = np.full(len(mean), np.std(mean) * 0.1)
            
            return mean, std
