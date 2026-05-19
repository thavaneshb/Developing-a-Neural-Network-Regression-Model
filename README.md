# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A neural network regression model is used to predict continuous numerical values based on input data. Unlike classification problems that assign inputs to categories, regression focuses on estimating real-valued outputs such as price, temperature, or demand.

The problem statement in this context involves developing a model that can learn the relationship between input features (independent variables) and a continuous target variable (dependent variable) using a neural network. The dataset typically contains multiple input attributes, and the goal is to train the model so that it can accurately predict the output for unseen data.

A neural network consists of layers of interconnected neurons, including an input layer, one or more hidden layers, and an output layer. Each neuron processes inputs using weights and biases, applies an activation function, and passes the result to the next layer. During training, the network adjusts these weights using optimization techniques like gradient descent to minimize the error between predicted and actual values.

In regression tasks, commonly used loss functions include Mean Squared Error (MSE) or Mean Absolute Error (MAE), which measure how far the predictions are from the true values. The model learns by iteratively updating its parameters to reduce this loss.

The main objective of this problem is to:

Build a neural network model

Train it using the dataset Evaluate its performance

Use it to make accurate continuous predictions

This approach is widely used in applications such as house price prediction, stock forecasting, and demand estimation, where outputs are numerical rather than categorical.

## Neural Network Model
Include the neural network model diagram.
<img width="1116" height="757" alt="image" src="https://github.com/user-attachments/assets/30a0a5da-dc5e-4c3c-83cb-7938aa495f47" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: ROGITH J

### Register Number:212224040280

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('DL1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name:AADHITHYAA L
# Register Number:212224220003
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,1)
        self.fc3=nn.Linear(1,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Name:AADHITHYAA L
# Register Number:212224220003
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
#Append loss inside the loop
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
Include screenshot of the generated data
<img width="432" height="414" alt="image" src="https://github.com/user-attachments/assets/c6e77102-18c3-48e6-b6a5-7ac933d6f689" />

### OUTPUT
<img width="766" height="248" alt="image" src="https://github.com/user-attachments/assets/0da72a50-fbc4-4693-8796-8a28d15ec5c6" />

### Training Loss Vs Iteration Plot
Include your plot here
<img width="782" height="568" alt="image" src="https://github.com/user-attachments/assets/c4d9e565-c6e7-40b2-a371-0e605df9dfd0" />

### New Sample Data Prediction
Include your sample input and output here
<img width="755" height="44" alt="image" src="https://github.com/user-attachments/assets/7040775b-d41e-4c18-aa37-d8787ccf82e7" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
