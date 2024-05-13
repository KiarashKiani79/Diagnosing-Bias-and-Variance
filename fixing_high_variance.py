from imports import *

#* The following dataset has 2 input_feature *#
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')
# x_train.shape = (60, 2) , y_train.shape = (60, )
# x_cv.shape = (20, 2) , y_cv.shape = (20, )

#! Try increasing the regularization parameter

# Define lambdas to plot
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params,
                            x_train, y_train,
                            x_cv, y_cv,
                            degree= 4,
                            baseline=250,
                            save_image=False,
                            image_name="improve your cross validation error by increasing the value of  𝜆")

