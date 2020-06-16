import pandas as pd
import webbrowser


def gen_df_from_json(df_json, X_col=None, y_col=None): 
    """ Import the pandas dataframe from json string

    Args:
        df_json (json): json string for dataframe
        X_col (str, optional): name of x-column. Defaults to None.
        y_col (str, optional): name of y-column. Defaults to None.

    Returns:
        DataFrame: if X_col and y_col are not specified
        list: containing DataFrame, x-column, and y-column
    """     
    df = pd.read_json(str(df_json))
    if X_col and y_col:
        x = df[X_col].values.reshape(-1, 1)
        y = df[y_col].values.reshape(-1, 1)
        return df, x, y
    else:
        return df
    
    
def open_browser():
    """ open web browser when running the app
    """    
    webbrowser.open_new('http://127.0.0.1:5000/')
