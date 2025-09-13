from abc import ABC, abstractmethod
import pandas as pd


# =====================================================================
# Abstract Base Class for Data Inspection Strategies
# =====================================================================
# This abstract class defines a common interface for all data inspection
# strategies. Every concrete strategy must implement the `inspect` method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# =====================================================================
# Concrete Strategy: Data Types and Non-null Counts
# =====================================================================
# Inspects the dataframe to show data types of each column and the count
# of non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Display data types and non-null counts for each column.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# =====================================================================
# Concrete Strategy: Data Shape
# =====================================================================
# Prints the overall shape (rows, columns) of the dataframe.
class DataShapeStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print the shape (number of rows, columns) of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe whose shape is inspected.

        Returns:
        None: Prints the shape of the dataframe.
        """
        print("\nData Shape:")
        print(df.shape)


# =====================================================================
# Concrete Strategy: Data Head
# =====================================================================
# Prints the first 5 rows of the dataframe for a quick preview.
class DataHeadStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print the first five rows of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the top 5 rows of the dataframe.
        """
        print("First Five Rows of Data:\n", df.head())


# =====================================================================
# Concrete Strategy: Data Tail
# =====================================================================
# Prints the last 5 rows of the dataframe for inspection.
class DataTailStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print the last five rows of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the last 5 rows of the dataframe.
        """
        print("Last Five Rows of Data:\n", df.tail())


# =====================================================================
# Concrete Strategy: Random Sample Rows
# =====================================================================
# Prints a random sample of rows from the dataframe (default = 5).
class DataRandomRowsStrategy(DataInspectionStrategy):
    def __init__(self, n=5):
        """
        Initialize the strategy.

        Parameters:
        n (int): Number of random rows to sample (default = 5).
        """
        self.n = n

    def inspect(self, df: pd.DataFrame):
        """
        Print a random sample of rows from the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints random sampled rows from the dataframe.
        """
        print(f"\nRandom {self.n} Rows from Dataset:")
        print(df.sample(n=self.n))


# =====================================================================
# Concrete Strategy: Duplicate Values
# =====================================================================
# Prints Duplicate values from data.
class DuplicateValuesStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print duplicate values from the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints duplicate values from the dataframe.

        """

        print("\nDuplicate Values:")
        print(df[df.duplicated()])
        duplicates = df[df.duplicated()]
        print(f"\nNumber of Duplicate Rows: {duplicates.shape[0]}")
        if not duplicates.empty:
            print("Sample Duplicate Rows:\n", duplicates.head())


# =====================================================================
# Concrete Strategy: Column Overview
# =====================================================================
# Prints Overview of the each column.
class ColumnOverviewStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print an overview of each column including its name,
        data type, number of unique values, and missing value count.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints column overview information.
        """
        overview = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes,
            "Missing Values (%)": (df.isnull().sum() / len(df)) * 100,
            "Unique Values": df.nunique(),
        })
        print("\nColumn Overview:")
        print(overview)


# =====================================================================
# Concrete Strategy: Summary Statistics
# =====================================================================
# Provides descriptive statistics for both numerical and categorical
# columns in the dataframe.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Print summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics for both numerical and categorical columns.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# =====================================================================
# Context Class: DataInspector
# =====================================================================
# The context class that uses a data inspection strategy.
# Allows switching strategies dynamically.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initialize the DataInspector with a specific strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to use.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Change the current inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to use.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Execute the current inspection strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection logic.
        """
        self._strategy.inspect(df)


# =====================================================================
# Example Usage
# =====================================================================
if __name__ == "__main__":
    # Example of using DataInspector with different strategies

    # Load your dataset
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize inspector with data types strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # Switch to summary statistics strategy
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass

__all__ = [
    'DataInspector',
    'DataInspectionStrategy',
    'DataTypesInspectionStrategy', 
    'SummaryStatisticsInspectionStrategy',
    'DataShapeStrategy',
    'DataHeadStrategy',
    'DataTailStrategy',
    'DataRandomRowsStrategy',
    'DuplicateValuesStrategy',
    'ColumnOverviewStrategy'
]
