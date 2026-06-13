import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("Salary_Data.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# -------------------------
# Data Visualization
# -------------------------

# Experience vs Salary
plt.scatter(data['YearsExperience'], data['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# Salary Distribution
sns.histplot(data['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# Prepare Data
# -------------------------

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Linear Regression
# -------------------------

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# -------------------------
# Decision Tree
# -------------------------

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

# -------------------------
# Random Forest
# -------------------------

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# -------------------------
# Accuracy Comparison
# -------------------------

lr_score = r2_score(y_test, lr_pred)
dt_score = r2_score(y_test, dt_pred)
rf_score = r2_score(y_test, rf_pred)

print("\nModel Accuracy:")
print("Linear Regression:", lr_score)
print("Decision Tree:", dt_score)
print("Random Forest:", rf_score)

# Accuracy Graph
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
scores = [lr_score, dt_score, rf_score]

plt.bar(models, scores)
plt.title("Model Comparison")
plt.ylabel("R2 Score")
plt.show()

# -------------------------
# Salary Prediction
# -------------------------

experience = [[5]]

predicted_salary = lr.predict(experience)

print("\nPredicted Salary for 5 Years Experience:")
print(predicted_salary[0])