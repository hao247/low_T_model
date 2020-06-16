import pandas as pd
from models.regression import perform_regression


def single_param_regression(X, y, pow_range, test_size=0.3, metric='MSE'):
    """ Performing linear regression of single-parameter model 
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
    for p in pow_range:
        X_with_power = X ** p
        model, error = perform_regression(X_with_power, y, test_size=test_size, metric=metric)
        if not best_error or (best_error > error and metric in {'MAE', 'MSE'}) or (best_error < error and metric == 'R2'):
            best_error, best_model, best_p = error, model, p
        errors.append([p, error])
    errors_df = pd.DataFrame(errors, columns=['Exponent', 'Error'])
    return {'best_model': [best_p, best_model, best_error], 'errors': errors_df}