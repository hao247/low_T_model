import pandas as pd
from models.regression import perform_regression


def single_param_regression(X, y, pow_range, test_size=0.3, metric='MSE'):
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