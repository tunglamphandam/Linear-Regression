#import vital library
import numpy as np 
import pandas as pd 
import matplotlib as plt 

iters = 100
learning_rate = 0.00001

#Prepare data
data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
area = [] 
price = []
for i in range(len(x)) :
    area.append(x[i][0])
    price.append(y[i][0])

#Function 
def cost_function(area, price , weight, bias):
    houses = len(area)
    total_error = 0.0
    for i in range(houses):
        total_error += (price[i] - (weight*area[i] + bias))**2
    return total_error / houses
def update_weights(area, price , weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    houses = len(area)

    for i in range(houses):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*area[i] * (price[i] - (weight*area[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(price[i] - (weight*area[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / houses) * learning_rate
    bias -= (bias_deriv / houses) * learning_rate

    return weight, bias
def train(area,price, weight, bias, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weight,bias = update_weights(area, price, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(area , price , weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))
    return weight, bias, cost_history

#Expression to calculate price prediction
def predict_price(area, price, bias):
    return weight*area + bias

train(area, price, weight, bias, learning, iters)

