from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

def plot_lr_residuals(model, X_test, y_test_log, outpath: Path):
    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual ($ actual - $ predicted)")
    plt.title("Linear Regression Residuals (test)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _feature_names(ct: ColumnTransformer):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                names.extend(list(trans.get_feature_names_out(cols)))
            except Exception:
                names.extend(list(cols))
        else:
            names.extend(list(cols))
    return names

def plot_tree_importance(tree_pipe, outpath: Path, top_k: int = 20):
    prep = tree_pipe.named_steps["prep"]
    model = tree_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = _feature_names(prep)
    importances = pd.Series(model.feature_importances_, index=names)
    top = importances.sort_values(ascending=False).head(top_k)[::-1]
    plt.figure()
    plt.barh(top.index, top.values)
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
