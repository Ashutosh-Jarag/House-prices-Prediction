from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import statsmodels
import numpy as np
from scipy import stats


class OutlierAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, numerical_columns : list):
        """
        Performs outlier detection analysis by identifying and visualizing outliers.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        numerical_columns (list): List of numerical columns to check for outliers.

        Returns:
        None
        """
        self.identify_outliers(df, numerical_columns)
        self.visualize_outliers(df, numerical_columns)


    @abstractmethod
    def identify_outliers(self, df: pd.DataFrame, numerical_columns: list):
        """
        Abstract method to identify outliers in the DataFrame.
        Concrete implementations should define their outlier identification logic here.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        numerical_columns (list): List of numerical columns to check for outliers.

        Returns:
        None

        """
        pass

    @abstractmethod
    def visualize_outliers(self, df: pd.DataFrame, numerical_columns: list):
        """
        Abstract method to visualize outliers in the DataFrame.
        Concrete implementations should define their outlier visualization logic here.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        numerical_columns (list): List of numerical columns to check for outliers.
        
        """
        pass




class IQROutlierAnalysis(OutlierAnalysisTemplate):
    def identify_outliers(self, df, numerical_columns):
        """
        Identifies outliers using (IQR)Interquartile Range method.
        Prints the number of outliers in each numerical column.

        parameters:
            df(pd.DataFrame): The dataframe to be analyzed.
            numerical_columns(list): List of numerical columns to check for outliers.

        Returns:
            None
        
        """    
        print("Identifying outliers using IQR method:")
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"{col}: Outerliers.shape[0] Outliers")

    def visualize_outliers(self, df, numerical_columns):
        """
        visualizes outliers using boxplots for each numerical column.
        """

        print("Visualizing outliers using boxplot:")
        plt.figure(figsize=(10,6))
        for col in numerical_columns:
            plt.boxplot(df[col])
            plt.title(col)
            plt.show()


        # print("\nVisualizing Outliers with Boxplots...")
        # plt.figure(figsize=(14, len(numerical_columns) * 4))
        # for i, col in enumerate(numerical_columns, 1):
        #     plt.subplot(len(numerical_columns), 1, i)
        #     sns.boxplot(x=df[col], color="skyblue")
        #     plt.title(f"Outliers in {col}")
        # plt.tight_layout()
        # plt.show()


class ZScoreOutlierAnalysis(OutlierAnalysisTemplate):
    def identify_outliers(self, df, numerical_columns):
            """
            Identifies outliers using z-score method.
            Prints the number of outliers in each numerical column.

            Parameters:
            df(pd.DataFrame): The dataframe to be analyzed.
            numerical_columns(list): List of numerical columns to check for outliers.

            Returns:
            None
            """
            print("Identifying outliers using Z-Score method:")
            for col in numerical_columns:
                z_scores = zscore(df[col].dropna())
                outliers = df[abs(z_scores) > 3]
                print(f"{col}: {outliers.shape[0]} Outliers")


    def visualize_outliers(self, df, numerical_columns):
        """
        Visulizing z-score outliers using boxplots and subplots for each numerical column.

        Parameters:
        df(pd.DataFrame): The dataframe to be analyzed.
        numerical_columns(list): List of numerical columns to check for outliers.

        """
        print("Visualizing outliers using boxplots...")
        plt.figure(figsize=(12, len(numerical_columns) * 4))
        for i, col in enumerate(numerical_columns, 1):
            plt.subplot(len(numerical_columns), 1, i)
            sns.boxplot(x=df[col], color="lightcoral")
            plt.title(f"Outliers in {col}")
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the OutlierAnalysisTemplate class.

    
    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Outlier Detection
    # outlier_detector = IQRBasedOutlierDetection()
    # outlier_detector.analyze(df, numerical_columns=["Age", "Salary"])

    #Or

    # outlier_detector = IQRBasedOutlierDetection()
    # outlier_detector.analyze(df, numerical_columns=[list_of_numerical_columns])    

    pass
