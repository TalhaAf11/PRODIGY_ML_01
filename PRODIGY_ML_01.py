import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
train_data = pd.read_csv('train.csv')  # Replace with your actual path
test_data = pd.read_csv('test.csv')   # Replace with your actual path

# Define features and target column
features = ['SquareFootage', 'Bedrooms', 'Bathrooms']  # Adjust these column names to match your dataset
target = 'Price'  # Replace with the actual target column name in train.csv

# Check the columns in the train DataFrame to ensure the specified columns exist
print(train_data.columns)

# Define the columns you want to check for NaN values
columns_to_check = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'Price']

# Filter the columns to only include those that exist in the DataFrame
existing_columns = [col for col in columns_to_check if col in train_data.columns]

# Drop rows with NaN values in the existing columns
train_data = train_data.dropna(subset=existing_columns)

# Optionally, print the resulting DataFrame to confirm
print(train_data)

# Handle missing values (if any)
train_data = train_data.dropna(subset=features + [target])
test_data = test_data.dropna(subset=features)

# Separate features (X) and target (y) from the training data
X = train_data[features]
y = train_data[target]

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation Mean Squared Error: {mse}")

# Predict on the test dataset
test_data['PredictedPrice'] = model.predict(test_data[features])

# Save predictions to a CSV file
test_data[['Id', 'PredictedPrice']].to_csv('predictions.csv', index=False)  # Replace 'Id' with the identifier column if different
print("Predictions saved to predictions.csv")
