#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Recruitment Pipeline Optimization for HR Analytics
# Using Manual Implementation Without Built-in Functions

# Sample Dataset
data = [
    {"EmpID": "RM297", "Age": 18, "AgeGroup": "18-25", "Attrition": "Yes", "BusinessTravel": "Travel_Rarely"},
    {"EmpID": "RM302", "Age": 18, "AgeGroup": "18-25", "Attrition": "No", "BusinessTravel": "Travel_Rarely"},
    {"EmpID": "RM458", "Age": 18, "AgeGroup": "18-25", "Attrition": "Yes", "BusinessTravel": "Travel_Frequently"},
    {"EmpID": "RM728", "Age": 18, "AgeGroup": "18-25", "Attrition": "No", "BusinessTravel": "Non-Travel"}
]

# Step 1: Encode Categorical Data Manually
def encode_categorical(data, column):
    unique_values = []
    for row in data:
        if row[column] not in unique_values:
            unique_values.append(row[column])
    
    for row in data:
        row[column] = unique_values.index(row[column])
    return unique_values

# Encode columns
attrition_labels = encode_categorical(data, "Attrition")
business_travel_labels = encode_categorical(data, "BusinessTravel")

# Step 2: Prepare Features and Labels
X = []  # Features
y = []  # Labels
for row in data:
    X.append([row["Age"], row["BusinessTravel"]])  # Features: Age and BusinessTravel
    y.append(row["Attrition"])                      # Label: Attrition

# Step 3: Implement a Simple Classification Algorithm (e.g., Perceptron)
def train_perceptron(X, y, epochs, learning_rate):
    weights = [0] * len(X[0])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute linear combination
            linear_combination = sum(X[i][j] * weights[j] for j in range(len(weights))) + bias
            
            # Apply step function
            prediction = 1 if linear_combination > 0 else 0

            # Calculate error
            error = y[i] - prediction

            # Update weights and bias
            for j in range(len(weights)):
                weights[j] += learning_rate * error * X[i][j]
            bias += learning_rate * error

    return weights, bias

# Convert y labels to binary
y = [1 if label == 0 else 0 for label in y]  # Yes -> 1, No -> 0

# Train the Perceptron model
weights, bias = train_perceptron(X, y, epochs=10, learning_rate=0.1)

# Step 4: Prediction
def predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        linear_combination = sum(X[i][j] * weights[j] for j in range(len(weights))) + bias
        predictions.append(1 if linear_combination > 0 else 0)
    return predictions

predictions = predict(X, weights, bias)

# Step 5: Evaluate the Model
correct = sum(1 for i in range(len(y)) if y[i] == predictions[i])
accuracy = correct / len(y)

# Output Results
print("Weights:", weights)
print("Bias:", bias)
print("Predictions:", predictions)
print("Accuracy:", accuracy)


# In[ ]:




