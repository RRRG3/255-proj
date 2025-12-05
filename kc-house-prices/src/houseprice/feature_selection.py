"""Feature selection utilities based on model importance."""
import pandas as pd
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline


def get_feature_importance(model: Pipeline, 
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model."""
    # Get the final estimator from pipeline
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
    else:
        estimator = model
    
    # Check if model has feature importances
    if not hasattr(estimator, "feature_importances_"):
        raise RuntimeError(
            "Model does not have feature_importances_ attribute. "
            "Ensure the model is fitted and supports feature importance."
        )
    
    importances = estimator.feature_importances_
    
    # Handle case where feature_names length doesn't match
    if len(feature_names) != len(importances):
        # Try to get feature names from preprocessor
        if hasattr(model, "named_steps") and "prep" in model.named_steps:
            try:
                feature_names = model.named_steps["prep"].get_feature_names_out()
            except:
                # If that fails, use generic names
                feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create dataframe and sort by importance
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return importance_df


def select_features_by_importance(X: pd.DataFrame, 
                                  importance_df: pd.DataFrame, 
                                  threshold: float) -> pd.DataFrame:
    """
    Select features based on importance threshold."""
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold}")
    
    # Get features above threshold
    selected_features = importance_df[
        importance_df["importance"] >= threshold
    ]["feature"].tolist()
    
    # Filter to features that exist in X
    selected_features = [f for f in selected_features if f in X.columns]
    
    if len(selected_features) == 0:
        # No features meet threshold, return all features with warning
        import warnings
        warnings.warn(
            f"No features meet importance threshold {threshold}. "
            "Returning all features."
        )
        return X.copy()
    
    return X[selected_features].copy()


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """Extract feature names from a fitted preprocessing pipeline."""
    if not hasattr(pipeline, "named_steps"):
        return []
    
    if "prep" not in pipeline.named_steps:
        return []
    
    prep = pipeline.named_steps["prep"]
    
    try:
        # Try to get feature names from preprocessor
        feature_names = prep.get_feature_names_out()
        return list(feature_names)
    except:
        # If that fails, return empty list
        return []
