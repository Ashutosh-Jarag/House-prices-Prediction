import pandas as pd

class ColumnTypeAnalyzer:
    """
    A utility class to divide dataframe columns into numerical and categorical.
    """

    @staticmethod
    def get_column_types(df: pd.DataFrame):
        """
        Splits dataframe columns into numerical and categorical.

        Parameters:
        df (pd.DataFrame): Input dataframe.

        Returns:
        tuple: (numerical_columns, categorical_columns)
        """
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\nNumerical Columns:")
        print(numerical_columns if numerical_columns else "None")

        print("\nCategorical Columns:")
        print(categorical_columns if categorical_columns else "None")

        return numerical_columns, categorical_columns
