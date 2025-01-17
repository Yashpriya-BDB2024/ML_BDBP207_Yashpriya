### https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html

from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

### EDA - Exploratory Data Analysis

# fetch_california_housing: A function from sklearn.datasets to download the California Housing dataset. 
# as_frame=True: This ensures that the dataset is returned as Pandas DataFrame
california_housing = fetch_california_housing(as_frame=True)
# .DESCR attribute: Contains the textual description/overview of the dataset
print(california_housing.DESCR)

# Displays the first 5 rows of the dataset's feature set (i.e., attributes of each sample, excluding the target variable)
print(california_housing.frame.head())

# Displays the first 5 rows of the feature data (without target variable - MedHouseVal) to check the str. & types of features.
print(california_housing.data.head())

# Shows the first 5 values of the target (MedHouseVal)  
print(california_housing.target.head())

# Will provide summary of dataset, i.e., data types, no. of samples (rows), no. of features (columns), and if the dataset contains any missing value (count of non-null values for each column)
california_housing.frame.info()

# .hist() - creates histogram for all the features in the dataset 
# figsize=(12, 10): Specifies the size of the figure (12 inches by 10 inches)
# bins=30: More bins allow finer granularity in the distribution
# edegecolor="black": Adds black edges to the histogram bars for better visibility
california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
# subplots_adjust(): hspace (space b/w rows), wspace (space b/w columns) - adjusts the spacing b/w subplots in the figure
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.plot()
# Displays the histogram
plt.show()

# Since, average rooms, average bedrooms, average occupation, and population has large range of the data with unnoticeable bin for the largest values.
# It means there are very high & few values (outliers)
# If in output, we see huge difference b/w max & 75% values, we can infer that there are a couple of extreme values present.
features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
print(california_housing.frame[features_of_interest].describe())

# Scatter plot - to decide if there are locations associated with high-valued houses
sns.scatterplot(
    data=california_housing.frame,
    x="Longitude",
    y="Latitude",
    size="MedHouseVal",
    hue="MedHouseVal",
    palette="viridis",
    alpha=0.5,
)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
_ = plt.title("Median house value depending of \n their spatial location")
plt.plot()
plt.show()

# Random subsampling - to have less data points to plot but that could still allow us to see these specificities
rng = np.random.RandomState(0)
indices = rng.choice(
    np.arange(california_housing.frame.shape[0]), size=500, replace=False
)
sns.scatterplot(
    data=california_housing.frame.iloc[indices],
    x="Longitude",
    y="Latitude",
    size="MedHouseVal",
    hue="MedHouseVal",
    palette="viridis",
    alpha=0.5
)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 1), loc="upper left")
_ = plt.title("Median house value depending of \n their spatial location")
plt.plot()
plt.show()

# Final analysis - pair plot of all features & the target but dropping the longitude & latitude
columns_drop=["Longitude", "Latitude"]    # Dropping the unwanted columns
subset = california_housing.frame.iloc[indices].drop(columns=columns_drop)
# Quantize the target & keep the midpoint for each interval
subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)
_ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")
plt.plot()
plt.show()

# linear predictive model
alphas = np.logspace(-3, 1, num=30)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
cv_results = cross_validate(
    model,
    california_housing.data,
    california_housing.target,
    return_estimator=True,
    n_jobs=2
)
score = cv_results["test_score"]
print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")

# Coefficient values obtained via cross-validation
# Latitude, longitude & median income are useful features in predicting the median house values
coefs = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns = california_housing.feature_names
)
color = {"whiskers": "black", "medians": "black", "caps": "black"}
coefs.plot.box(vert=False, color=color)
plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
_ = plt.title("Coefficients of Ridge models\n via cross-validation")
plt.plot()
plt.show()
