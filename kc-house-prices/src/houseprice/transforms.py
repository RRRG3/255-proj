"""Target variable transformation utilities"""
import numpy as np
from scipy import stats
from typing import Optional


class TargetTransformer:
    """Handle target variable transformations for regression tasks"""
    
    def __init__(self, method: str = "log1p"):
        valid_methods = ["log1p", "boxcox", "yeojohnson", "none"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}"
            )
        
        self.method = method
        self.lambda_: Optional[float] = None
        self.fitted_ = False
        
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).flatten()
        
        if self.method == "none":
            self.fitted_ = True
            return y
            
        elif self.method == "log1p":
            self.fitted_ = True
            return np.log1p(y)
            
        elif self.method == "boxcox":
            # Box-Cox requires strictly positive values
            if np.any(y <= 0):
                raise ValueError(
                    "Box-Cox transformation requires strictly positive values. "
                    "Consider using 'yeojohnson' or 'log1p' instead."
                )
            transformed, self.lambda_ = stats.boxcox(y)
            self.fitted_ = True
            return transformed
            
        elif self.method == "yeojohnson":
            transformed, self.lambda_ = stats.yeojohnson(y)
            self.fitted_ = True
            return transformed
            
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(
                "Transformer must be fitted before inverse_transform. "
                "Call fit_transform first."
            )
        
        y_transformed = np.asarray(y_transformed).flatten()
        
        if self.method == "none":
            return y_transformed
            
        elif self.method == "log1p":
            return np.expm1(y_transformed)
            
        elif self.method == "boxcox":
            return stats.inv_boxcox(y_transformed, self.lambda_)
            
        elif self.method == "yeojohnson":
            # Inverse Yeo-Johnson transformation
            return self._inv_yeojohnson(y_transformed, self.lambda_)
    
    def _inv_yeojohnson(self, y: np.ndarray, lmbda: float) -> np.ndarray:
        y = np.asarray(y)
        out = np.zeros_like(y)
        
        # For y >= 0
        pos_mask = y >= 0
        if lmbda == 0:
            out[pos_mask] = np.expm1(y[pos_mask])
        else:
            out[pos_mask] = np.power(y[pos_mask] * lmbda + 1, 1 / lmbda) - 1
        
        # For y < 0
        neg_mask = ~pos_mask
        if lmbda == 2:
            out[neg_mask] = -np.expm1(-y[neg_mask])
        else:
            out[neg_mask] = 1 - np.power(-(2 - lmbda) * y[neg_mask] + 1, 1 / (2 - lmbda))
        
        return out
