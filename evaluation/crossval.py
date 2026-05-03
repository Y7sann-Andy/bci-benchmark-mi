from __future__ import annotations                  
                                                                                                                                                                                              
from typing import Any, Callable                                                                                                                                                              
                                                                                                                                                                                              
import numpy as np                                                                                                                                                                            
from sklearn.metrics import accuracy_score                                       
from sklearn.model_selection import StratifiedKFold                                                                                                                                           
 
                                                                                                                                                                                              
def cross_validate_within_session(                                               
    pipeline_factory: Callable[[], Any],
    X: np.ndarray,                     
    y: np.ndarray,                              
    n_splits: int = 5,                              
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    random_state: int | None = 42,                                                                                                                                                            
    verbose: bool = False,                      
) -> dict[str, Any]:                                                                                                                                                                          
    """Stratified K-fold CV on within-session trials.                            
                                                                                                                                                                                              
    Parameters                                  
    ----------                                                                                                                                                                                
    pipeline_factory : callable () -> estimator                                  
        Zero-arg callable returning a FRESH pipeline each fold. Must expose                                                                                                                   
        `.fit(X, y)` and `.predict(X) -> y_pred`. Factory (not instance)
        prevents cross-fold leakage.                                                                                                                                                          
    X : ndarray of shape (n_trials, n_channels, n_times)                         
    y : ndarray of shape (n_trials,), integer labels starting at 0.
    n_splits : int                              
        Number of folds. Default 5.                                                                                                                                                           
    metric : callable (y_true, y_pred) -> float
        Default accuracy.                                                                                                                                                                     
    random_state : int or None                                                   
        For reproducible fold assignment. Default 42.                                                                                                                                         
    verbose : bool                                                                                                                                                                            
        If True, print per-fold score as folds complete.
                                                                                                                                                                                              
    Returns                                                                      
    -------                                                                                                                                                                                   
    dict with keys:
        'scores' : list[float]   per-fold scores in fold order                                                                                                                                
        'mean'   : float                                                         
        'std'    : float                                                                                                                                                                      
        'y_true' : ndarray       concatenated true labels (fold order)           
        'y_pred' : ndarray       concatenated predictions
        'test_idx' : ndarray     concatenated indices into original X (fold order) 
    """

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores, y_true_all, y_pred_all, test_idx_all = [], [], [], []

    for train_idx, test_idx in splitter.split(X, y):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        score = metric(y[test_idx], y_pred)
        scores.append(score)                                                                                                                            
        y_true_all.append(y[test_idx])
        y_pred_all.append(y_pred)
        test_idx_all.append(test_idx)
        if verbose:
            print(f"Fold {len(scores)}/{n_splits} score: {score:.4f}")                                                    

    return {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'y_true': np.concatenate(y_true_all),
        'y_pred': np.concatenate(y_pred_all),
        'test_idx': np.concatenate(test_idx_all),
    }              