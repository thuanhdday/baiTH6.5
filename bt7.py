import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('Data_Number_7.csv')

# Tạo chỉ số "nguy cơ biến chứng"
df['risk_score'] = (df['bmi'] * 0.3) + (df['blood_glucose'] * 0.5) + (df['hospitalizations'] * 0.2)

# Tạo nhóm tuổi
bins = [0, 40, 60, float('inf')]
labels = ['<40', '40-60', '>60']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Kiểm định chi-squared giữa nhóm tuổi và biến chứng
contingency_table = pd.crosstab(df['age_group'], df['complication'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"p-value for chi-squared test: {p}")

# Tạo đặc trưng "xu hướng đường huyết"
df['blood_glucose_trend'] = df['blood_glucose'].diff()
df['blood_glucose_trend'] = df['blood_glucose_trend'].apply(
    lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'Stable')
)
df['blood_glucose_trend'].fillna('Stable', inplace=True)  # giá trị đầu tiên bị NaN sau diff

# Tạo đặc trưng "mức độ nghiêm trọng"
df['severity'] = (df['hospitalizations'] * 0.5) + (df['blood_glucose'] * 0.5)

# Tập đặc trưng và nhãn
X = df[['bmi', 'blood_glucose', 'hospitalizations', 'age_group', 'risk_score', 'blood_glucose_trend', 'severity']]
X = pd.get_dummies(X, drop_first=True)
y = df['complication']

# Chia dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, log_reg_pred))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Report:\n", classification_report(y_test, rf_pred))

# Tối ưu siêu tham số Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search.best_params_}")

# SMOTE xử lý mất cân bằng
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Huấn luyện lại Random Forest sau SMOTE
rf_smote = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf_smote.fit(X_res, y_res)
rf_res_pred = rf_smote.predict(X_test)
print("Random Forest (SMOTE) Report:\n", classification_report(y_test, rf_res_pred))
