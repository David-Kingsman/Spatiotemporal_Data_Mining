"""树模型baseline：RF、XGBoost、LightGBM"""
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
    """树模型配置"""
    model_type: Literal["rf", "xgb", "lgbm"] = "xgb"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    quantile_regression: bool = False  # 是否使用分位数回归
    quantiles: list = None  # 分位数列表，如 [0.1, 0.5, 0.9]


class TreeBaseline:
    """树模型baseline"""
    
    def __init__(self, config: TreeConfig):
        self.config = config
        self.model = None
        self.quantile_models = {}  # 用于分位数回归
        
        if config.quantiles is None:
            config.quantiles = [0.1, 0.5, 0.9]
        
        # 如果指定的模型不可用，自动降级
        if config.model_type == "xgb" and not XGBOOST_AVAILABLE:
            print("警告: XGBoost不可用，自动切换到Random Forest")
            config.model_type = "rf"
        elif config.model_type == "lgbm" and not LIGHTGBM_AVAILABLE:
            print("警告: LightGBM不可用，自动切换到Random Forest")
            config.model_type = "rf"
    
    def _create_model(self, quantile: Optional[float] = None):
        """创建模型实例"""
        if self.config.model_type == "rf":
            if not SKLEARN_AVAILABLE:
                raise ImportError("需要安装 scikit-learn")
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.config.model_type == "xgb":
            if not XGBOOST_AVAILABLE:
                raise ImportError("需要安装 xgboost")
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
                raise ImportError("需要安装 lightgbm")
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
            raise ValueError(f"未知的模型类型: {self.config.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        if self.config.quantile_regression:
            # 为每个分位数训练一个模型
            for q in self.config.quantiles:
                model = self._create_model(quantile=q)
                model.fit(X, y)
                self.quantile_models[q] = model
        else:
            # 标准回归
            self.model = self._create_model()
            self.model.fit(X, y)
            # 对于RF，计算训练误差用于不确定性估计
            if self.config.model_type == "rf":
                y_pred_train = self.model.predict(X)
                self.train_std = np.std(y - y_pred_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测均值"""
        if self.config.quantile_regression:
            # 使用中位数作为预测
            if 0.5 in self.quantile_models:
                return self.quantile_models[0.5].predict(X)
            else:
                # 如果没有0.5，使用平均值
                predictions = [model.predict(X) for model in self.quantile_models.values()]
                return np.mean(predictions, axis=0)
        else:
            return self.model.predict(X)
    
    def predict_quantiles(self, X: np.ndarray) -> dict:
        """预测分位数"""
        if not self.config.quantile_regression:
            raise ValueError("需要启用 quantile_regression=True")
        
        results = {}
        for q, model in self.quantile_models.items():
            results[q] = model.predict(X)
        return results
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测均值和标准差（从分位数估计）
        
        返回:
        mean: 预测均值
        std: 估计的标准差
        """
        if self.config.quantile_regression:
            quantiles = self.predict_quantiles(X)
            # 使用分位数估计均值和标准差
            if 0.5 in quantiles:
                mean = quantiles[0.5]
            else:
                mean = np.mean([q for q in quantiles.values()], axis=0)
            
            # 从分位数估计标准差（假设正态分布）
            if 0.1 in quantiles and 0.9 in quantiles:
                # 使用10%和90%分位数估计标准差
                std = (quantiles[0.9] - quantiles[0.1]) / (2 * 1.28)  # 1.28是90%区间的z值
            elif 0.25 in quantiles and 0.75 in quantiles:
                std = (quantiles[0.75] - quantiles[0.25]) / (2 * 0.675)  # IQR
            else:
                # 使用所有分位数的范围估计
                all_quantiles = np.array(list(quantiles.values()))
                std = np.std(all_quantiles, axis=0)
            
            return mean, std
        else:
            # 对于非分位数模型，使用经验估计
            mean = self.predict(X)
            
            # 对于RF，可以使用所有树的预测的标准差作为不确定性
            if self.config.model_type == "rf" and hasattr(self.model, 'estimators_'):
                # 获取所有树的预测
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                std = np.std(tree_predictions, axis=0)
            # 使用训练误差的标准差
            elif hasattr(self, 'train_std'):
                std = np.full(len(mean), self.train_std)
            else:
                # 启发式估计：使用预测值的一定比例
                std = np.full(len(mean), np.std(mean) * 0.1)
            
            return mean, std

