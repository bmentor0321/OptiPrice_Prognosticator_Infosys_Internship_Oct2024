def get_mean(df, column):
    """
    description
    Params:
    df (pd.DataFrame): desc
    column (str): desc

    Returns:
    mean (float): desc

    """
    
    all_columns = list(df.columns)
    # assert column in all_columns
    try:
        mean = df[column].mean()
        return mean
    except:
        return "Please check the column name"