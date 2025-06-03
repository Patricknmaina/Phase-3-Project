# This file contains functions for different stages in the modeling process.
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# -----------------Data Preparation-------------------!
# function to check for null and duplicate values, and handle them
def clean_nulls_and_duplicates(df):
    """
    This function cleans a dataframe by checking for, and handling null values and duplicate rows.
    It also standardizes the columns by removing the whitespaces between the words, adding a hyphen for readability and capitalizing the first letter in each word.

    Parameters:
        df(pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with no duplicate or null values, and standardized columns
    """

    print("Initial shape of the dataset:", df.shape)

    # check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nNull values detected in the following columns:")
        print(null_counts[null_counts > 0])
        
        # drop the missing values if any
        df = df.dropna(axis=0)
        print("Dropped rows with missing values")
    else:
        print("\nNo null values detected.")

    # check for duplicate values
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} duplicate rows.")
        
        # drop the duplicate rows
        df = df.drop_duplicates()
        print("Dropped duplicate rows.")
    else:
        print("\nNo duplicate rows detected.")

    # Standardize the column names
    df.columns = (
        df.columns
        .str.strip()                          # Remove leading/trailing whitespace
        .str.title()                          # Capitalize first letter of each word
        .str.replace(' ', '_', regex=False)   # Replace spaces with hyphens
    )

    print(df.columns)

    print("\n Final shape of data:", df.shape)
    return df


# ---------------Exploratory Data Analysis-------------------!
# function to plot categorical features for univariate analysis
def categorical_distributions(df, feature):
    """
    This function will plot the distribution of a categorical feature on a given dataframe.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The desired column from the dataframe
    """

    # plot the distribution
    plt.figure(figsize=(14, 5))
    sns.countplot(x=feature, data=df, palette='deep', order=df[feature].value_counts().index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# function to plot the distribution of numerical features for univariate analysis
def numerical_distribution(df, numerical_features):
    """
    Plots distribution plots with KDE curves for a list of numerical features in the given dataframe

    Parameters:
        df(pd.DataFrame): the input dataframe containing the numerical features
        numerical_features: list of column names to plot
    """

    # calculate the subplot grid size
    no_of_rows = (len(numerical_features) - 1) // 3 + 1
    no_of_cols = min(3, len(numerical_features))

    # create subplots
    fig, axes = plt.subplots(nrows=no_of_rows, ncols=no_of_cols, figsize=(16, 4 * no_of_rows))
    axes = axes.flatten() if len(numerical_features) > 1 else [axes]

    # plot each numerical feature
    for n, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[n], color='blue', edgecolor='black')
        axes[n].set_title(f"Distribution of {feature}", fontsize=10)
        axes[n].set_xlabel(feature)
        axes[n].set_ylabel('Count')

    # omit any unused subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    # improve layout spacing
    fig.tight_layout()
    plt.show()


# function to plot categorical features vs Churn for bivariate analysis
def categorical_churn(df, feature):
    """
    This function plots the distribution of a categorical feature, with churn as a comparative variable

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The categorical column to investigate
    """

    # plot the distribution
    plt.figure(figsize=(10, 5))
    churn_count = df.groupby(feature)['Churn'].sum().sort_values(ascending=False)
    top_10_categories = churn_count.head(10).index.tolist()
    sns.countplot(x=feature, hue='Churn', data=df, order=top_10_categories)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.show()


# function to plot numerical columns with the churn rate for bivariate analysis.
def kde_plots_with_churn(df, feature, type_of_charge):
    """
    This function plots the distribution of the numerical features based on the churn rate.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The numerical feature to plot
        type_of_charge: the specific charge type(day, evening, night, international)
    """

    # kde plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=feature, hue='Churn', fill=True)
    plt.xlabel(f"Total {type_of_charge} Charge")
    plt.ylabel("Density")
    plt.title(f"Churn Distribution by Total {type_of_charge} Charges")
    plt.show()

# function to plot the correlation matrix for feature correlation with target variable
def correlation_heatmap(df):
    """
    This function plots a correlation heatmap that illustrates the correlation between the numerical features and the target(Churn)

    Parameters:
        df(pd.DataFrame): The input dataframe
    """

    # define plot size
    plt.figure(figsize=(14, 14))

    # compute the correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    # create a mask that will hide the upper triangle
    mask = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))

    # plot the heatmap
    sns.heatmap(
        data=mask,
        cmap='magma',
        annot=True,
        fmt=".1g",
        vmin=-1
    )

    # define the title and display plot
    plt.title('Feature Correlatiom Heatmap', fontsize=16)
    plt.show()