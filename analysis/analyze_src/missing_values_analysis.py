from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method should create a visualization (e.g., heatmap) of missing values.
        """
        pass


# Concrete Class for Missing Values Identification
# -------------------------------------------------
# This class implements methods to identify and visualize missing values in the dataframe.
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


class MissingvaluePerPercentage(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the percentage of missing values for each column in the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values percentage to the console.
        
        """
        print("\nMissing Values Percentage by Column:")
        print(((df.isnull().sum()/len(df))*100).sort_values(ascending=False))

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a bar plot to visualize the missing values percentage in the dataframe.
        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.
        Returns:
        None: Displays a bar plot of missing values percentage.
        
        """
        print("\nVisualizing Missing Values Percentage...")
        plt.figure(figsize=(12, 6))
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_percentage.sort_values(ascending=False).head(5).plot(kind='bar', color='skyblue')

        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.title('Missing Values Percentage per Column')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  

class ColumnWithToManyNulls(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Print the columns with more than 70% null values.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        
        Returns:
        None: Prints the columns with more than 70% null values.
        """
        print("\nColumns with More Than 70% Null Values:")
        null_percentages = (df.isnull().sum() / len(df)) * 100
        print(null_percentages[null_percentages>70])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a Horizontal bar plot to visualize the columns with more than 70% null values.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a horizontal bar plot of columns with more than 70% null values.

        """
        print("\nVisualizing Columns with More Than 70% Null Values...")
        plt.figure(figsize=(12, 6))
        null_percentages = (df.isnull().sum() / len(df)) * 100
        columns_with_many_nulls = null_percentages[null_percentages > 70]
        columns_with_many_nulls.sort_values(inplace=True)
        columns_with_many_nulls.plot(kind='barh', color='skyblue')
        plt.xlabel('Missing Percentage (%)')
        plt.ylabel('Columns')
        plt.title('Columns with More Than 70% Null Values')
        plt.tight_layout()
        plt.show()




# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass
