# Linear Regression Model

This is a simple implementation of a linear regression model using numpy and matplotlib. The model is capable of fitting a linear function to a given set of training data, and predicting new values based on the learned parameters.

## Installation

To use this model, you will need to have numpy, matplotlib and sklearn installed. You can install these libraries using pip:
```
pip install numpy matplotlib sklearn
```

## Usage

To use this model, you can create an instance of the LinearRegression class, passing in the training data and an initial set of parameters:

```python
# Import packages
from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

if __name__ == "__main__":
    # Create training data
    x,y = make_regression(n_samples=100, n_features=1, noise=10)
    x = np.hstack((x, np.ones(x.shape)))
    y = y.reshape(y.shape[0], 1)
    
    # Initialize parameters
    theta = np.random.randn(2,1)

    # Create model and fit to data
    LinearRegression1 = LinearRegression(x, y, theta)
    print(LinearRegression1.theta)
    LinearRegression1.gradient_descent(theta, x, y, 0.01, 1000)
    print(LinearRegression1.theta)

    # Predict new values
    print(LinearRegression1.predict(x))

    # Show learning curve
    LinearRegression1.show_learning_curve()

    # Plot predictions
    LinearRegression1.show_model(0)
```

## Contributing
If you have suggestions for how this model could be improved, or want to report a bug, please open an issue or a pull request.
