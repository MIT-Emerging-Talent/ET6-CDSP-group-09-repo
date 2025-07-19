# **Description Cell 1: Notebook Objective**

**Description:**  
This notebook serves as the first step in the data analysis pipeline.  
Its primary purpose is to load the raw dataset (`clean_merged_data.csv`),  
perform a thorough data quality audit, engineer new features essential for  
the analysis (such as lags, rolling averages, and interaction terms), and  
then export the cleaned and enriched data to a new file named  
`analysis_ready_data.csv`. This new file will be the single source of truth  
for all subsequent analysis notebooks.

## **Code Cell 1: Setup and Configuration**

**Description:**  
This cell imports all the necessary Python libraries and defines the file  
paths for the input and output data. Centralizing this configuration makes the  
notebook easier to manage and adapt. We use the `pathlib` library for robust  
and operating-system-agnostic path handling.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Define file paths
# This assumes the notebook is in the '4_data_analysis' folder
# and the data is in the '0_datasets' folder.
input_file = Path("../1_datasets/final_datasets/clean_merged_data.csv")
output_file = Path("../4_data_analysis/output_data/analysis_ready_data.csv")

# Ensure the output directory exists
output_file.parent.mkdir(parents=True, exist_ok=True)
```

### **Code Cell 2: Data Loading and Initial Inspection**

**Description:**  
This cell loads the dataset from the specified input file and performs a  
high-level initial inspection. We check the dataset's dimensions (number of  
rows and columns), list the column names, and display the first few rows to  
get a feel for the data's structure and content.

```python
# Load the dataset
df = pd.read_csv(input_file)

# Display basic information
print(f"Dataset loaded from: {input_file}")
print("--- Initial Inspection ---")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
display(df.head())
```

### **Code Cell 3: Data Quality Audit**

**Description:**  
This cell conducts a detailed data quality assessment to identify potential  
issues. We check the data types of each column, count the number of missing  
values, check for and count any duplicate rows, and verify that the 'Year'  
column falls within our expected range (2010-2019).

```python
# Perform a detailed data quality audit
print("--- Data Quality Audit ---")

# Check data types and non-null counts
print("\nData Types and Non-Null Counts:")
df.info()

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Check for duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows found: {num_duplicates}")

# Check for years outside the expected range
invalid_years_count = df[
    ~df['Year'].between(2010, 2019)
].shape[0]
print(
    f"Number of rows with Year outside 2010–2019: {invalid_years_count}"
)
```

### **Code Cell 4: Feature Engineering**

**Description:**  
In this crucial step, we engineer new features to enhance the dataset  
for modeling. This includes:

1. **Sorting** the data by Country and Year to ensure correct time-series calculations.
2. Creating **lag features** (`PM25_lag1`, `PM25_lag2`) to capture the effect  
   of past pollution.
3. Creating **rolling average features** (`PM25_3yr_avg`, `PM25_5yr_avg`)  
   to model cumulative exposure.
4. Creating an **interaction term** between PM₂.₅ and SDI to test if
   development level modifies the pollution-health relationship.
5. Creating a categorical variable for **SDI** to enable stratified analysis  
   in later notebooks.

```python
# --- Feature Engineering ---
print("Starting feature engineering...")

# Ensure data is sorted for time-series operations
df = df.sort_values(['Country', 'Year']).reset_index(drop=True)

# 1. Create 1-year and 2-year lag features for PM2.5
df['PM25_lag1'] = df.groupby('Country')['PM2.5'].shift(1)
df['PM25_lag2'] = df.groupby('Country')['PM2.5'].shift(2)

# 2. Compute 3-year and 5-year rolling averages for PM2.5
df['PM25_3yr_avg'] = (
    df.groupby('Country')['PM2.5']
      .rolling(window=3, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)
df['PM25_5yr_avg'] = (
    df.groupby('Country')['PM2.5']
      .rolling(window=5, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

# 3. Create interaction term: PM2.5 * SDI
df['PM25_SDI_interaction'] = df['PM2.5'] * df['SDI']

# 4. Create SDI categories for stratification
sdi_bins = [0.0, 0.45, 0.61, 0.75, 1.0]
sdi_labels = ['Low', 'Medium', 'High', 'Very High']
df['SDI_category'] = pd.cut(
    df['SDI'],
    bins=sdi_bins,
    labels=sdi_labels,
    include_lowest=True
)

print("Feature engineering complete.")

# Verify the new features by displaying a sample
print("\nSample of DataFrame with new features:")
new_features = [
    'PM25_lag1', 'PM25_lag2', 'PM25_3yr_avg', 'PM25_5yr_avg',
    'PM25_SDI_interaction', 'SDI_category'
]
display(df[['Country', 'Year', 'PM2.5', 'SDI'] + new_features].head(7))
```

### **Code Cell 5: Export Processed Data**

**Description:**  
This final cell saves the fully cleaned and feature-enriched DataFrame  
to a new CSV file. This `analysis_ready_data.csv` file will serve as  
the input for all subsequent analytical notebooks, ensuring consistency  
and reproducibility.

```python
# Export the processed DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print("--- Export Complete ---")
print(
    f"Processed data with {df.shape[1]} columns saved to: {output_file}"
)
print(f"Final shape of the analysis-ready data: {df.shape}")
```
