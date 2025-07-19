# Data Analysis Workflow: PM₂.₅ Health Impact Study

This directory contains the notebooks and scripts used to analyze the
relationship between PM₂.₅, socio‑demographic development (SDI), and
health outcomes (DALYs). The analysis is broken down into a series of
focused, sequential Jupyter notebooks. This modular structure enhances
clarity, reproducibility, and ease of collaboration.

Each notebook has a specific objective and relies on data artifacts
created by the preceding notebooks in the workflow.

## Modular Notebook Structure

- **01_data_quality_prep.ipynb**  
  - Purpose: Loads `clean_merged_data.csv`, audits data quality,  
    engineers features (lags, rolling averages), and saves a clean dataset.  
  - Key Outputs:  
    - `analysis_ready.csv`  
    - Data quality summary

- **02_exploratory_trends.ipynb**  
  - Purpose: Generates descriptive statistics, distribution plots,  
    temporal trends, and geographic maps for key variables.  
  - Key Outputs:  
    - PNG/SVG figures in `../6_figures`  
    - `country_level_averages.csv`

- **03_modeling_dalys.ipynb**  
  - Purpose: Trains and evaluates Linear Regression and Random Forest  
    models for each DALY outcome; logs metrics and saves best models.  
  - Key Outputs:  
    - `*.pkl` models in `../5_model_artifacts`  
    - `model_performance_scores.csv`

- **04_sdi_vulnerability.ipynb**  
  - Purpose: Conducts SDI‑stratified modeling, feature importance analysis,  
    and policy‑relevant threshold and sensitivity checks.  
  - Key Outputs:  
    - Figures in `vulnerability_plots/` folder  
    - `sdi_rf_metrics.csv`

- **05_summary_report.ipynb**  
  - Purpose: Imports metrics and figures to auto‑generate a final,  
    comprehensive summary as Markdown/PDF.  
  - Key Outputs:  
    - Markdown/PDF export of the final report

## Notebook Workflow Outline

### 1. `01_data_quality_prep.ipynb`

**Objective:** To create a clean, analysis‑ready dataset.  
**Input:** `../0_datasets/clean_merged_data.csv`  
**Steps:**

- Load and preview the raw dataset.
- Perform data quality checks:
  - Missing values
  - Duplicates
  - Valid year range (2010–2019)
- Engineer new features:
  - `PM25_lag1`, `PM25_lag2`
  - `PM25_3yr_avg`, `PM25_5yr_avg`
  - `PM25_SDI_interaction`
  - `SDI_category`
- Export the cleaned and enriched DataFrame.

**Output:** `../0_datasets/analysis_ready.csv`

### 2. `02_exploratory_trends.ipynb`

**Objective:** To visualize and understand the data’s characteristics.  
**Input:** `../0_datasets/analysis_ready.csv`  
**Steps:**

- Generate distribution plots (histograms, boxplots) for PM₂.₅, SDI, and DALYs.
- Create faceted time‑series plots showing 2010–2019 trends by SDI.
- Plot a choropleth map of PM₂.₅ and All‑cause DALYs.
- Compute and visualize a correlation heatmap.
- Save all figures and a CSV of country‑level averages.

**Output:**

- Figures in `../6_figures`
- `../0_datasets/country_level_averages.csv`

### 3. `03_modeling_dalys.ipynb`

**Objective:** To train and evaluate predictive models for each outcome.  
**Input:** `../0_datasets/analysis_ready.csv`  
**Steps:**

- Define predictor and target variables.
- Split data into training (80%) and testing (20%) sets.
- Loop through each DALY outcome to train:
  - Linear Regression
  - Random Forest
- Evaluate models using R² and RMSE.
- Save scores and pickle the best Random Forest models.
- Generate and save residual diagnostic plots.

**Output:**

- `*.pkl` models in `../5_model_artifacts`
- `model_performance_scores.csv`

### 4. `04_sdi_vulnerability.ipynb`

**Objective:** To investigate how socio‑demographic context modifies the
pollution‑health relationship.  
**Input:** `../0_datasets/analysis_ready.csv` and pickled models.  
**Steps:**

- Build separate Random Forest models for each SDI category.
- Compare R² and feature importances across SDI strata.
- Conduct threshold analysis using median PM₂.₅.
- Perform sensitivity analysis with reduced feature subsets.

**Output:**

- Figures in `vulnerability_plots/`
- `sdi_rf_metrics.csv`

### 5. `05_summary_report.ipynb`

**Objective:** To synthesize all results into a final report.  
**Input:** All previously generated metrics, figures, and models.  
**Steps:**

- Load all saved analytical artifacts.
- Compose markdown cells narrating findings from each stage.
- Dynamically display key tables and figures.
- Discuss policy implications and project limitations.
