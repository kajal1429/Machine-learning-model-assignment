# Machine-learning-model-assignment-
problem statement- build models to identify issues at Fleet level, vehicle level, to identify average vehicle performance, divide into 10 buckets of performance, and rank vehicles which are under performing across different parameteres.

Data Analysis
Step 1: Data Preparation
Data Cleaning and Transformation: first we need to Ensure the data is cleaned of any outliers or missing values. Transform the data if necessary (e.g., scaling, normalization).

Feature Engineering: Create relevant features from the existing data that might help in better understanding vehicle performance and issues (e.g., derived features like power consumption, efficiency metrics).

Step 2: Exploratory Data Analysis (EDA)
Descriptive Statistics: Understand the distribution and summary statistics of each parameter (RPM, Current, Throttle Command, Motor Temperature, Battery Temperature).

Correlation Analysis: Identify relationships between parameters (e.g., how RPM affects Motor Temperature, Current affects Battery Temperature).

Visualization: Plot histograms, scatter plots, and heatmaps to visualize distributions and correlations. This helps in identifying patterns and potential issues (like clusters of vehicles with similar problems).

Machine Learning Model Development
Step 3: Model Building
Fleet Level Performance Analysis:

Regression Models: Predict fleet-level average performance metrics (e.g., average RPM, average Motor Temperature).
Clustering: Using algorithms like K-means clustering to group vehicles into 5 clusters based on their performance metrics derived from the data.
Identifying Under-Performing Vehicles:

Classification Models: Build classifiers to identify vehicles under different performance categories (overload, normal load, underload).
Anomaly Detection: Utilize techniques such as Isolation Forest or One-Class SVM to detect outliers that might indicate vehicles with abnormal performance characteristics.

Step 4: Predictive Maintenance
Monitoring Motor Degradation:
Train models to predict motor degradation using historical data on RPM, Current, and Motor Temperature. This could involve time-series analysis or regression models.
Battery Health Monitoring:

Develop models to predict battery temperature trends based on Current and Ambient Temperature data. Identify batteries at risk of overheating or degradation.

Step 5: Model Deployment and Monitoring
Scalability: Ensure models are scalable to handle new data from additional vehicles.
Monitoring: Implement a monitoring system to track model performance over time and retrain models periodically with new data to maintain accuracy.

Implementation Steps
Data Collection: Continuously collect data from vehicles to update the models and improve accuracy over time.
Model Evaluation: Use metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
Deployment: Integrate the models into a system that can classify new vehicles based on their performance metrics and issue alerts for vehicles requiring attention.

Conclusion- we can effectively build models to analyze fleet performance , classify vehicles based on performance metrics, and implement predictive maintenance strategies.
Import libraries and load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

 load and prepare data
 Assuming data is stored in a CSV file
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

Interpretation and deployment
# Print model coefficients to understand feature importance
coefficients = model.coef_
intercept = model.intercept_

print('Model Coefficients:')
for feature, coef in zip(X.columns, coefficients):
    print(f'{feature}: {coef}')

print(f'Intercept: {intercept}')







