import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from pydantic import BaseModel
    return BaseModel, mo, np, pd, plt, sns, stats


@app.cell
def _(mo):
    mo.md(r"""# Functions""")
    return


@app.cell
def _(mo):
    mo.md("""## Statistics Functions""")
    return


@app.cell
def _(BaseModel, np, pd, stats):
    class CHI2Result(BaseModel):
        chi2_stat: float
        p_value: float
        significant: bool

    def perform_chi2_test(observed, expected=None, correction=True, alpha=0.05) -> CHI2Result:
        """
        Performs a Chi-squared test and handles common issues.

        Args:
            observed (array-like or pandas.Series): Observed frequencies/counts.
            expected (array-like or pandas.Series, optional): Expected frequencies/counts.
                                                             If None, assumed to be equal. Defaults to None.
            correction (bool, optional): Use Yates' correction for continuity.
                                         Defaults to True.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
           CHI2Result: (chi2_statistic, p_value, significant) or (0.0, 0.0, False) if test can't be performed.
        """
        try:
            # Convert to numpy arrays for easier handling
            observed = np.array(observed)

            if expected is not None:
                expected = np.array(expected)

            # Check data types and convert to integers if necessary.
            if not np.issubdtype(observed.dtype, np.integer):
                observed = observed.astype(int)
            if expected is not None and not np.issubdtype(expected.dtype, np.integer):
                expected = expected.astype(int)

            # Handle edge cases
            if (observed < 0).any():
                raise ValueError("Observed frequencies must be non-negative")
            if expected is not None and (expected <= 0).any():
                raise ValueError("Expected frequencies must be positive")

            # Check if expected is provided, else, the expected data is generated equally distributed.
            if expected is None:
                expected = np.full(observed.shape, np.sum(observed) / observed.size, dtype=np.int32)
                print("expected values are not provided, generated with equal distribution.")

            if observed.shape != expected.shape:
                raise ValueError("Observed and expected arrays must have the same shape.")

            # Minimum requirement: at least two categories
            if len(observed) < 2:
                raise ValueError("At least two categories are required for a chi-squared test.")

            # Check for zero expected values
            if (expected == 0).any():
                raise ValueError("Expected frequencies should not be zero. Cannot perform test.")

            # Check minimum expected frequency condition
            if (expected < 5).any():
                print("Warning: Some expected values are less than 5. Chi-squared test might be unreliable.")

            # If all requirements are met, perform the Chi-Squared test
            chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected, ddof=0 if correction else 0)

            significant = p_val < alpha

            return CHI2Result(chi2_stat=chi2_stat, p_value=p_val, significant=significant)

        except ValueError as e:
            print(f"Error: {e}. Cannot perform chi-squared test.")
            return CHI2Result(chi2_stat=0.0, p_value=0.0, significant=False)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return CHI2Result(chi2_stat=0.0, p_value=0.0, significant=False)

    class TTestResult(BaseModel):
        t_stat: float
        p_value: float
        significant: bool

    def perform_t_test(sample1, sample2, paired=False, equal_var=True, alpha=0.05) -> TTestResult:
        """
        Performs a t-test and handles common issues.

        Args:
            sample1 (array-like or pandas.Series): First sample data.
            sample2 (array-like or pandas.Series): Second sample data.
            paired (bool, optional):  If True, perform paired t-test. Defaults to False.
            equal_var (bool, optional): If True (independent samples), assume equal population variances. Defaults to True.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            TTestResult(None, None, False): (t_statistic, p_value, signficant) or (None, None, False) if test can't be performed.
        """
        MINIMUM_SAMPLES = 2

        try:
            # Convert to numpy arrays for easier handling
            sample1 = np.array(sample1)
            sample2 = np.array(sample2)

            # Check data types and convert to float if needed
            if not np.issubdtype(sample1.dtype, np.number) or not np.issubdtype(sample2.dtype, np.number):
                sample1 = sample1.astype(float)
                sample2 = sample2.astype(float)

            # Check for non-numeric data
            if not np.all(np.isfinite(sample1)) or not np.all(np.isfinite(sample2)):
                raise ValueError("Data contains non-numeric values (e.g. NaN, inf). Cannot perform test")

            # Handle the case with no variance
            if np.var(sample1) == 0 or np.var(sample2) == 0:
                raise ValueError("One of the groups have 0 variance, cannot perform t-test")

            # Check sample sizes
            if len(sample1) < MINIMUM_SAMPLES or len(sample2) < MINIMUM_SAMPLES:
                raise ValueError("Each sample must have at least 2 data points for a t-test.")

            # Check for paired test requirements if needed
            if paired and len(sample1) != len(sample2):
                raise ValueError("For a paired t-test, samples must have the same length.")

            # Handle the variance test for independent sample
            if not paired and equal_var == False:
                var1 = np.var(sample1, ddof=1)
                var2 = np.var(sample2, ddof=1)

                if var1 / var2 > 4 or var2 / var1 > 4:
                    print("Variance of the samples are very different, consider setting 'equal_var' parameter to false.")

            # Perform the appropriate t-test
            if paired:
                t_stat, p_val = stats.ttest_rel(sample1, sample2)
            else:
                t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=equal_var)

            significant = p_val < alpha

            return TTestResult(t_stat=t_stat, p_value=p_val, significant=significant)

        except ValueError as e:
            print(f"Error: {e}. Cannot perform t-test.")
            return TTestResult(t_stat=0.0, p_value=0.0, significant=False)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return TTestResult(t_stat=0.0, p_value=0.0, significant=False)


    def statistical_analysis(df, alpha=0.05):
        """
        Analyzes the readmission data using statistical tests.

        Args:
          df (pandas.DataFrame): Input dataframe.
          alpha (float): Significance Level
        """
        print("----- Starting Analysis -----")

        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        # Remove Readmitted from the list to avoid errors
        if "Readmitted" in categorical_features:
          categorical_features.remove("Readmitted")

        print("\n### Chi-squared Tests ###")
        for feature in categorical_features:
            print(f"\n* Readmitted vs. {feature}:")
            contingency_table = pd.crosstab(df["Readmitted"], df[feature])
            chi2_result = perform_chi2_test(contingency_table.values)
            if chi2_result.significant:
                print(f"{feature} is Significantly Correlated with Readmitted.")
            else:
                print(f"{feature} is NOT Significantly Correlated with Readmitted.")
            print(f"T-statistic: {chi2_result.chi2_stat:.2f}, p-value: {chi2_result.p_value:.3f}")

        print("\n### T-Tests ###")
        for feature in numerical_features:
            if feature != "Patient_ID":  # Skip Patient_ID as it's not a meaningful numerical feature to test
                print(f"\n* {feature} by Readmission status:")
                group_readmitted = df[df["Readmitted"] == "Yes"][feature].dropna()
                group_not_readmitted = df[df["Readmitted"] == "No"][feature].dropna()
                ttest_result = perform_t_test(group_readmitted, group_not_readmitted)
                if ttest_result.significant:
                    print(f"{feature} is Significantly Correlated with Readmitted.")
                else:
                    print(f"{feature} is NOT Significantly Correlated with Readmitted.")
                print(f"T-statistic: {ttest_result.t_stat:.2f}, p-value: {ttest_result.p_value:.3f}")



        print("\n----- Analysis Complete -----")
    return (
        CHI2Result,
        TTestResult,
        perform_chi2_test,
        perform_t_test,
        statistical_analysis,
    )


@app.cell
def _(mo):
    mo.md("""## Graphing Functions""")
    return


@app.cell
def _(pd, plt):
    def box_plot_of_numerical_values(df: pd.DataFrame, numerical_cols: list[str]) -> None:
        # Box plot to showcase the distribution and outliers for numerical columns
        plt.figure(figsize=(10, 6))
        data_to_plot = [df[col] for col in numerical_cols]
        plt.boxplot(data_to_plot,
                    tick_labels=numerical_cols,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops = dict(color='red'))
        plt.title("Distribution and Outliers for Numerical Columns", fontsize=14)
        plt.ylabel("Values", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    return (box_plot_of_numerical_values,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data Exploration
        Lets load our dataset and get a basic picture of what we are looking at, as well as removing useless columns.
        """
    )
    return


@app.cell
def _(pd):
    # Load the data
    data_filename = '../data/hospital_readmissions.parquet'
    df = pd.read_parquet(data_filename)

    # drop useless field
    drop_cols = [
        "Patient_ID"
    ]
    df = df.drop(drop_cols, axis=1)

    df.head(5)
    return data_filename, df, drop_cols


@app.cell
def _(df, np):
    target_col = "Readmitted"

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()

    print(f"Categorical Features: {categorical_features}")
    print(f"Numerical Features: {numerical_features}")
    return categorical_features, numerical_features, target_col


@app.cell
def _(mo):
    mo.md(r"""Next we will run a basic `df.describe` to get some basic statistics on our data.""")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""And here is a box plot of our data to better visual outliers and the data we see above:""")
    return


@app.cell
def _(box_plot_of_numerical_values, df, numerical_features):
    box_plot_of_numerical_values(df, numerical_features)
    return


@app.cell
def _(mo):
    mo.md(r"""Woah, I see that outlier for the Age column. Lets take a closer look""")
    return


@app.cell
def _(df):
    # check on the outliers
    df[df['Age'] > 100]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Here we notice the outliers for Age. Specifically, there is 10 rows that are the age of 150. A quick google search says the oldest person to ever live was 122 years old. So I am inclined to believe that this is a data error. I could set these "error" ages to something like the mean of the age column, but since it is only 10 rows (1% of the total data), I will remove them from the dataset when I do training so they don't affect the model.

        Now, lets take a look at any missing data that exists:
        """
    )
    return


@app.cell
def _(df):
    # Check for missing values
    df.isnull().sum().sort_values(ascending=False)
    return


@app.cell
def _(df):
    # We see there is only missing values for A1C_Result, so let's explore that
    df["A1C_Result"].value_counts()
    return


@app.cell
def _(mo):
    mo.md("""Here we have quite a few missing values for A1C_Result (434). Rather than remove them from our dataset, we will fill in the null values with "Missing" so we can use that as a potential feature when training our model.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Statistical Analysis""")
    return


@app.cell
def _(df, statistical_analysis):
    statistical_analysis(df)
    return


@app.cell
def _(mo):
    mo.md(
        """
        **Significantly Correlated Features with Readmitted**

        * Num_Diagnoses by Readmission status (T-statistic: 2.13, p-value: 0.033)

        Here we are preforming Chi-Squared tests for categorical features and T-Tests for numerical features. We notice only one feature Num_Diagnoses is significantly correlated with the outcome column. This is potentially worrying for the model training as there might not be a lot of features that can help the model learn.

        Lets take a look at the correlation matrix.
        """
    )
    return


@app.cell
def _(df, numerical_features, plt, sns):
    # Correlation matrix for numerical features
    corr_matrix = df[numerical_features].corr()
    # plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Display the correlation values
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()
    return (corr_matrix,)


@app.cell
def _(df, numerical_features):
    # Correlation with the target variable
    corr_with_target = df[numerical_features].corrwith(df['Readmitted'].apply(lambda x: 1 if x == 'Yes' else 0))
    print("\nCorrelation with Readmitted:")
    print(corr_with_target)
    return (corr_with_target,)


@app.cell
def _(mo):
    mo.md("""Hmm not a huge amount of correlation from this matrix either. This is definitely worrying.""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Results
        - Some error data in Age needs to be removed before training
        - There is not a lot of correlation between the features and the target model. This might mean that model training will go poorly. We could potentially do some feature engineering to help with this.
        - Since we don't have a clear idea of what could be the best model to train on, I definitely think using something like PyCaret or Azure ML Studio's AutoML to explore many models could be worth it to find a decent starting point for our target model.
        """
    )
    return


if __name__ == "__main__":
    app.run()
