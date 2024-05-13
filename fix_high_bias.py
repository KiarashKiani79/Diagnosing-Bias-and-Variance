from imports import *

#* The following dataset has 1 input_feature *#
# Load and split the data
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data1.csv')
# x_train.shape = (60, 1) , y_train.shape = (60, )
# x_cv.shape = (20, 1) , y_cv.shape = (20, )

model = LinearRegression()
#* Adding polynomial features
#* Feature scaling
#* Training the model
#* Compute the training / cross validation MSE
#* Plotting the model
#* Save the plot image
utils.train_plot_poly(model, 
                      x_train, y_train,
                      x_cv, y_cv, 
                      max_degree=10, 
                      baseline=250,
                      save_image=False)

#! Try getting additional features
#* The following dataset has 2 input_feature *#
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')
# x_train.shape = (60, 2) , y_train.shape = (60, )
# x_cv.shape = (20, 2) , y_cv.shape = (20, )

model = LinearRegression()
utils.train_plot_poly(model, 
                      x_train, y_train,
                      x_cv, y_cv,
                      max_degree=6,
                      baseline=250,
                      image_name='data2_poly.png',
                      save_image=False)

#! Try decreasing the regularization parameter
# Define lambdas to plot
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]

# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, 
                            x_train, y_train,
                            x_cv, y_cv,
                            degree= 4,
                            baseline=250,
                            save_image=False,
                            image_name="decrease the value of  𝜆")
