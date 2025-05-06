import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

import shap
import optuna
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer

# Đọc dữ liệu
housing = pd.read_csv('HousingData.csv')
target_col = 'MEDV'
# Vẽ pairplot với 5 đặc trưng tương quan cao nhất day
top_features = housing.corr()[target_col].abs().sort_values(ascending=False).index[1:6]
sns.pairplot(housing[top_features.tolist() + [target_col]])
plt.show()

# In hệ số tương quan Pearson
print(housing.corr()[target_col].sort_values(ascending=False))

# Xử lý ngoại lai với Isolation Forest
features = housing.drop(columns=[target_col])
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(features)
housing = housing[outliers != -1].reset_index(drop=True)

# Kiểm tra VIF với xử lý NaN/inf
X_vif = housing.select_dtypes(include=[np.number]).drop(columns=[target_col]).copy()
X_vif = X_vif.replace([np.inf, -np.inf], np.nan)
X_vif = X_vif.dropna(axis=1)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False))

# Tạo feature tùy điều kiện (bạn có thể điều chỉnh tên biến phù hợp nếu cần)
if 'RM' in housing.columns and 'CRIM' in housing.columns:
    housing['room_per_crime'] = housing['RM'] / (housing['CRIM'] + 1e-5)

if 'TAX' in housing.columns:
    housing['high_tax'] = (housing['TAX'] > housing['TAX'].mean()).astype(int)

if set(['LSTAT', 'AGE']).issubset(housing.columns):
    housing['lstat_age'] = housing['LSTAT'] * housing['AGE']

# Phân chia đầu vào và đầu ra
X = housing.drop(columns=[target_col])
y = housing[target_col]

# Xử lý NaN trước khi tạo Polynomial features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_num = pd.DataFrame(X_imputed, columns=X.columns).select_dtypes(include=[np.number])
X_poly = poly.fit_transform(X_num)
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X_num.columns))

# Neural network model
def create_nn():
    model = Sequential()
    model.add(Dense(64, input_dim=X_poly.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Linear Regression
lr = LinearRegression()
lr.fit(X_scaled, y)

# Tối ưu XGBoost với Optuna
def objective(trial):
    model = XGBRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=42
    )
    score = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

xgb_best = XGBRegressor(**study.best_params)
xgb_best.fit(X_scaled, y)

# Mạng neural
nn_model = KerasRegressor(model=create_nn, epochs=50, batch_size=32, verbose=0)
nn_model.fit(X_scaled, y)

# Stacking
stack = StackingRegressor(
    estimators=[('xgb', xgb_best), ('nn', nn_model)],
    final_estimator=LinearRegression()
)
stack.fit(X_scaled, y)

# Đánh giá mô hình
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(model):
    mse = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error').mean()
    rmse = np.sqrt(mse)
    r2 = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2').mean()
    mape = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_percentage_error').mean()
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")

print("Đánh giá mô hình Stacking:")
evaluate(stack)

# Residual plot
y_pred = stack.predict(X_scaled)
residuals = y - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot - Stacking")
plt.show()

# SHAP
explainer = shap.Explainer(xgb_best)
shap_values = explainer(X_scaled)
shap.summary_plot(shap_values, features=X_poly, feature_names=X_poly.columns)
