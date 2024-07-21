
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming data is stored in a CSV file
data = pd.read_csv('vehicle_data.csv')

# Split data into features (X) and target (y)
X = data[['RPM', 'AC Motor Controller Current', 'Throttle Command']]
y = data['Motor Temperature']

# Convert categorical variables to numerical (if necessary)
# Example: Convert RPM from categorical bins to numerical
X['RPM'] = X['RPM'].astype('category').cat.codes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
2. Model Building
Train a Linear Regression Model
python
Copy code
# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
3. Interpretation and Deployment
Interpret Model Coefficients
python
Copy code
# Print model coefficients to understand feature importance
coefficients = model.coef_
intercept = model.intercept_

print('Model Coefficients:')
for feature, coef in zip(X.columns, coefficients):
    print(f'{feature}: {coef}')

print(f'Intercept: {intercept}')