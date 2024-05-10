from imports import *

# Load and split the data
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data1.csv')
# x_train.shape = (60, 1) , y_train.shape = (60, )
# x_cv.shape = (20, 1) , y_cv.shape = (20, )

# Initialize the linear-model
model = LinearRegression()

# Adding polynomial features
