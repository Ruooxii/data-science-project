# House Sales in King County, USA — Price Prediction

A complete data analysis workflow (importing → wrangling → EDA → modeling → evaluation) built in Jupyter Notebook as the capstone Final Project of CISC7204 Assignment 01.

## 1. Problem / Research Question

Can we predict the market price of a house in King County (which includes Seattle) from its features — such as living area, overall grade, and waterfront view — and how much can polynomial features and Ridge regularization improve that prediction?

## 2. Dataset

- **Source**: House sale prices in King County, USA (homes sold between May 2014 and May 2015), originally published on [Kaggle — House Sales in King County](https://www.kaggle.com/harlfoxem/housesalesprediction), slightly modified for the CISC7204 course.
- **Usage permissions**: Publicly available dataset, used here for educational purposes only.
- **Overview**: 21,613 house sales × 21 attributes, including `price` (target), `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, and `sqft_lot15`.

## 3. Methods & Tools

- **Language & tools**: Python 3, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, scikit-learn.
- **Analysis workflow**:
  1. **Importing data** — load the CSV with `pandas.read_csv`, inspect column dtypes and summary statistics (`describe()`).
  2. **Data wrangling** — drop the identifier column, check for and handle missing values in `bedrooms` and `bathrooms`.
  3. **Exploratory data analysis** — `value_counts` on floor levels, a Seaborn boxplot of price by waterfront view, a regression plot of `sqft_above` vs. price, and correlation analysis to rank all features against price.
  4. **Model development** — simple linear regression on `sqft_living`, multiple linear regression on 11 features, and a scikit-learn `Pipeline` (StandardScaler + PolynomialFeatures + LinearRegression).
  5. **Model evaluation & refinement** — train/test split, Ridge regression (alpha = 0.1), and a second-order polynomial Ridge model.

## 4. Key Findings & Limitation

**Key findings**
1. `sqft_living` is the strongest single predictor of price (r ≈ 0.70), followed by `grade` (≈ 0.67) and `sqft_above` (≈ 0.61). As the boxplot shows, waterfront homes have a much higher median price and far fewer high-price outliers than non-waterfront homes (see `waterfront_price_boxplot.png`).
2. Prediction accuracy improves step by step: a simple linear model on `sqft_living` reaches R² ≈ 0.49; a multiple linear model on 11 features reaches R² ≈ 0.66; adding polynomial features through a pipeline reaches R² ≈ 0.75 on training data. On held-out test data, Ridge regression (alpha = 0.1) achieves R² ≈ 0.66, and a second-order polynomial Ridge model achieves R² ≈ 0.70 (see `sqft_above_price_regplot.png`).

**Limitation**
- Longitude alone has essentially no predictive power (R² ≈ 0.0005), and even the best model leaves roughly 30% of price variance unexplained. The linear and polynomial models do not capture location effects well (zipcode is nearly uncorrelated with price in this encoding), and the data covers only one county during 2014–2015, so the models may not generalize to other markets or time periods.

## 5. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ruooxii/data-science-project.git
   cd data-science-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "CISC7204 Assgn01 Module 06 - FinalProj MC564835_YANG RUOXI.ipynb"
   ```
4. **Important**: in the notebook (Module 1), set the `filepath` variable to point to the included dataset, e.g.:
   ```python
   filepath = 'housing.csv'
   ```
   Then run all cells (Kernel → Restart & Run All).

## 6. Files in this repository

- `house_price_prediction.ipynb` — main analysis notebook
- `housing.csv` — King County house sales dataset (modified course version)
- `requirements.txt` — Python dependencies
- `waterfront_price_boxplot.png` — boxplot of price by waterfront view
- `sqft_above_price_regplot.png` — regression plot of `sqft_above` vs. price
