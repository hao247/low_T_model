from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from params import metrics


def perform_regression(X, y, test_size=0.3, metric='MSE'):
    metric_func = metrics[metric]
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return [model, metric_func(y_test, predictions)]