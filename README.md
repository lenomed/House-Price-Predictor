# House Price Predictor

A machine learning-based application that predicts house prices based on key features such as location, size, number of rooms, and amenities. This project demonstrates end-to-end machine learning workflow including data exploration, preprocessing, model training, and evaluation using Linear Regression.

## 📋 Project Overview

This project implements a complete ML pipeline to predict residential house prices. The model analyzes various property features including physical characteristics (area, bedrooms, bathrooms, stories) and amenities (parking, air conditioning, hot water heating, basement, guest room, etc.) to accurately estimate property values.

## 🎯 Features

- **Comprehensive Data Exploration**: Exploratory Data Analysis (EDA) with visualizations using Seaborn and Plotly
- **Data Preprocessing**: Categorical encoding and feature scaling for optimal model performance
- **Linear Regression Model**: Trained on normalized features for accurate price predictions
- **Performance Metrics**: Evaluation using MSE, RMSE, MAE, and R² Score
- **Correlation Analysis**: Feature importance ranking to understand price drivers

## 📊 Dataset

**File**: `Housing.csv`

**Target Variable**: `price` - House price (continuous value)

**Features** (13 total):
- **Numerical**: area, bedrooms, bathrooms, stories, parking
- **Categorical**: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus

**Data Shape**: Multiple housing records with complete feature set

## 🛠️ Technology Stack

### Core Libraries
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning algorithms and preprocessing
  - LinearRegression
  - train_test_split
  - StandardScaler
  - Metrics (MSE, RMSE, MAE, R²)

### Visualization
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations

## 📁 Project Structure

```
House-Price-Predictor/
├── machine/
│   └── main.py              # Main ML pipeline script
├── Housing.csv              # Training dataset
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lenomed/House-Price-Predictor.git
   cd House-Price-Predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly
   ```

   Or create a `requirements.txt` file with:
   ```
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   plotly
   ```

   Then install:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Project

Execute the main script from the project root directory:

```bash
python machine/main.py
```

### Expected Output

The script will display:
- Dataset information and statistics
- Data exploration summaries (head, info, describe)
- Null value counts
- Column names
- Feature correlation with price (sorted)
- Group-wise price analysis by hot water heating
- **Model Performance Metrics**:
  ```
  MSE : [value]
  RMSE: [value]
  MAE : [value]
  R²  : [value]
  ```

## 🔄 Workflow

The `main.py` script follows this pipeline:

### 1. **Data Loading & Exploration**
   ```python
   df = pd.read_csv('Housing.csv')
   print(df.head(), df.info(), df.describe())
   ```

### 2. **Feature Preprocessing**
   - Encode binary categorical variables (yes/no → 1/0):
     - prefarea
     - mainroad
     - guestroom
     - basement
     - hotwaterheating
     - airconditioning
   - One-hot encode multi-class categorical variable:
     - furnishingstatus (drop_first=True to avoid multicollinearity)

### 3. **Data Analysis**
   - Calculate correlation with target variable (price)
   - Analyze mean prices by amenity presence
   - Generate count plots for categorical features

### 4. **Data Splitting**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
   ```
   - 60% training data
   - 40% testing data

### 5. **Feature Scaling**
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

### 6. **Model Training**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

### 7. **Model Evaluation**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score (coefficient of determination)

## 📈 Model Performance

The Linear Regression model evaluates performance using:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error - Average of squared differences between predicted and actual values |
| **RMSE** | Root Mean Squared Error - Square root of MSE, in same units as target variable |
| **MAE** | Mean Absolute Error - Average absolute difference between predicted and actual values |
| **R²** | Coefficient of Determination - Proportion of variance explained by the model (0-1, higher is better) |

## 🔑 Key Insights

Based on the code's feature correlation analysis:
- Strong predictors are identified and ranked
- Categorical features are properly encoded to binary/one-hot format
- Model baseline uses all available features after preprocessing
- Standard scaling ensures uniform feature contribution during training

## 📝 Data Notes

- **CSV Path**: Currently hardcoded in main.py (adjust path as needed)
- **Missing Values**: Handled during data exploration phase
- **Categorical Encoding**: 
  - Binary features: Simple mapping (yes=1, no=0)
  - Multi-class: One-hot encoding with drop_first=True

## 🔧 Customization

### Modify Train-Test Split
```python
# Change test_size parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

### Add Visualization
Uncomment the visualization sections in main.py:
```python
# sns.pairplot(df, hue='price', palette='viridis')
# plt.show()
# sns.scatterplot(x='price', y='area', data=df)
# plt.show()
```

### Try Different Models
Replace LinearRegression with:
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
model = RandomForestRegressor()  # or GradientBoostingRegressor()
```

## 🎓 Learning Path

This project teaches:
1. Data loading and exploration with Pandas
2. Categorical variable encoding techniques
3. Feature scaling and normalization
4. Train-test data splitting
5. Model training and fitting
6. Regression model evaluation
7. Performance metric interpretation

## 📚 Dependencies Details

``` 
pandas>=1.0.0         # Data manipulation
numpy>=1.18.0         # Numerical operations
scikit-learn>=0.24.0  # ML algorithms & metrics
matplotlib>=3.0.0     # Visualization
seaborn>=0.11.0       # Statistical visualization
plotly>=4.0.0         # Interactive plots
```

## 🐛 Troubleshooting

### FileNotFoundError for Housing.csv
- Update the file path in the script to match your local setup
- Use relative path: `df = pd.read_csv('Housing.csv')`
- Ensure Housing.csv is in the project root directory

### Import Errors
```bash
pip install --upgrade pandas numpy scikit-learn matplotlib seaborn plotly
```

### StandardScaler Not Scaling Test Data Correctly
- Ensure `fit_transform()` is used only on training data
- Use `transform()` for test data (as shown in the code)

## 🚀 Future Enhancements

- [ ] Save trained model as pickle file for reuse
- [ ] Create web interface (Flask/Django)
- [ ] Add cross-validation for better evaluation
- [ ] Implement hyperparameter tuning
- [ ] Try ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add feature importance analysis
- [ ] Create prediction API
- [ ] Add interactive dashboard with Streamlit/Dash
- [ ] Convert script to modular functions/classes

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

**lenomed** - [GitHub Profile](https://github.com/lenomed)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Improve documentation

## 📞 Support

For issues or questions:
- Open an issue on [GitHub Issues](https://github.com/lenomed/House-Price-Predictor/issues)
- Check existing issues for solutions
- Review code comments and documentation

## 📖 References

- [scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Feature Scaling Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [One-Hot Encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

---

**Happy Predicting! 🏠📊**