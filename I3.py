import numpy as np
import pandas as pd

#noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
#x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
#N = x.shape[0]
#y = 15*x + 8 + 20*noise
#plt.scatter(x, y)
#Load data từ csv file
data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
numofPoint = N
area = data[:, 0].reshape(-1, 1)
price = data[:, 1].reshape(-1, 1)

#Cần tìm các tham số ( bias, weights) trước
def find_corr_x_y(x,y):                                         #1
    n = len(x)                                                  #2
    prod = []
    for xi,yi in zip(x,y):                                      #3
         prod.append(xi*yi)
         
    sum_prod_x_y = sum(prod)                                    #4
    
    sum_x = sum(x)
    sum_y = sum(y)
    
    squared_sum_x = sum_x**2
    squared_sum_y = sum_y**2 
    
    x_square = []
    for xi in x:
        x_square.append(xi**2)            
    x_square_sum = sum(x_square)
    y_square=[]
    for yi in y:
        y_square.append(yi**2)        
    y_square_sum = sum(y_square)
    
    # Use formula to calculate correlation                      #5
    numerator = n*sum_prod_x_y - sum_x*sum_y
    denominator_term1 = n*x_square_sum - squared_sum_x
    denominator_term2 = n*y_square_sum - squared_sum_y
    denominator = (denominator_term1*denominator_term2)**0.5
    correlation = numerator/denominator
    
    return correlation 
weight = find_corr_x_y(area, price)*np.std(price)/np.std(area) 
bias = np.mean(price) - weight * np.mean(area)
print(weight, bias)

#Input ra giá nhà dự đoán
def predict_price(weight, bias, area): 
    predict_price_1 = weight * area + bias
    return predict_price_1

