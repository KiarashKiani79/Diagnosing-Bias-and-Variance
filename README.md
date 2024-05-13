# Polynomial Regression Analysis

This repository contains Python scripts for analyzing and addressing high bias and high variance issues in polynomial regression models. The scripts utilize various techniques such as feature engineering, regularization, and model evaluation to improve the performance of polynomial regression models.

## File Structure

- `fix_high_bias.py`: Script to address high bias in polynomial regression models.
- `fixing_high_variance.py`: Script to address high variance in polynomial regression models.
- `utils.py`: Utility functions for data preprocessing, model training, and plotting.

## Usage

### Fixing High Bias (`fix_high_bias.py`)

This script addresses high bias by enhancing the complexity of the model using polynomial features, adjusting regularization parameters, and evaluating model performance.

1. **Load and Split Data**: Loads the dataset and splits it into training, cross-validation, and test sets.
2. **Model Training**: Trains a linear regression model on the training data.
3. **Feature Engineering**: Adds polynomial features to the input data.
4. **Model Evaluation**: Computes the training and cross-validation mean squared error (MSE) to assess model performance.
5. **Plotting**: Generates plots to visualize the model and the impact of polynomial degree and regularization parameters.
6. **Saving Results**: Optionally saves the generated plots for future reference.

### Fixing High Variance (`fixing_high_variance.py`)

This script tackles high variance by adjusting regularization parameters to prevent overfitting and improve generalization.

1. **Load and Split Data**: Loads the dataset and splits it into training, cross-validation, and test sets.
2. **Model Training**: Trains a linear regression model on the training data.
3. **Regularization**: Adjusts regularization parameters to control model complexity and prevent overfitting.
4. **Model Evaluation**: Computes the training and cross-validation mean squared error (MSE) to assess model performance.
5. **Plotting**: Generates plots to visualize the impact of regularization parameters on model performance.
6. **Saving Results**: Optionally saves the generated plots for future reference.

## Dataset

The scripts utilize datasets (`data/c2w3_lab2_data1.csv` and `data/c2w3_lab2_data2.csv`) for training and evaluation. These datasets contain features and corresponding target values for polynomial regression analysis.


## Images

![Degree of Polynomial vs. Train and CV MSEs](https://github.com/KiarashKiani79/Diagnosing-Bias-and-Variance/blob/main/plot_images/degree%20of%20polynomial%20vs.%20train%20and%20CV%20MSEs.png)
<div align="center">
  <p><em>Degree of Polynomial vs. Train and CV MSEs</em></p>
</div>

![Degree of Polynomial vs. Train and CV MSEs (Another Dataset)](https://github.com/KiarashKiani79/Diagnosing-Bias-and-Variance/blob/main/plot_images/data2_poly.png)
<div align="center">
  <p><em>Degree of Polynomial vs. Train and CV MSEs (Another Dataset)</em></p>
</div>

![Decrease the Value of 𝜆 Plot](https://github.com/KiarashKiani79/Diagnosing-Bias-and-Variance/blob/main/plot_images/decrease%20the%20value%20of%20%20%F0%9D%9C%86.png)
<div align="center">
  <p><em>Decrease the Value of 𝜆 Plot</em></p>
</div>

![Improve Your Cross Validation Error by Increasing the Value of 𝜆 Plot](https://github.com/KiarashKiani79/Diagnosing-Bias-and-Variance/blob/main/plot_images/improve%20your%20cross%20validation%20error%20by%20increasing%20the%20value%20of%20%20%F0%9D%9C%86.png)
<div align="center">
  <p><em>Improve Your Cross Validation Error by Increasing the Value of 𝜆 Plot</em></p>
</div>

