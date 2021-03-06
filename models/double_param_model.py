import pandas as pd
from models.regression import perform_regression


def double_param_regression(X, y, pow_range, test_size=0.3, metric='MSE'):
    """ Performing linear regression of double-parameter model 
    in a selected expoenent range

    Args:
        X (DataFrame): input dataframe
        y (DataFrame): labels
        pow_range (list): minimum and maximum of exponent 
        test_size (float, optional): Defaults to 0.3.
        metric (str, optional): Defaults to 'MSE'.

    Returns:
        dict: dict containing
        1. best model parameters
        2. errors of all exponent combinations
    """    
    errors = []
    best_error = None
    best_model = None
    for p1 in pow_range:
        for p2 in pow_range:
            X_with_power = pd.DataFrame([X.flatten() ** p1, X.flatten() ** p2]).T
            model, error = perform_regression(X_with_power, y, test_size=test_size, metric=metric)
            if not best_error or (best_error > error and metric in {'MAE', 'MSE'}) or (best_error < error and metric == 'R2'):
                best_error, best_model, best_p = error, model, [p1, p2]
            errors.append([p1, p2, error])
        errors_df = pd.DataFrame(errors, columns=['P1', 'P2', 'Error'])
    return {'best_model': [*best_p, best_model, best_error], 'errors': errors_df}
