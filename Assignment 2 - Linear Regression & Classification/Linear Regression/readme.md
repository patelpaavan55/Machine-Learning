Homework \#2 Programming Assignment[¶](#Homework-#2-Programming-Assignment)
===========================================================================

CSCI567, FALL 2019
Victor Adamchik
**Due: 11:59 pm, October 13, 2019**

### Before you start:[¶](#Before-you-start:)

There is a known issue with vocareum that caused by vocareum website, not us. If you submit your homework twice in a row without refresh the page, the grade book score might not be updated. So I suggest you to refresh your page before doing another submission.
 Also, this project have two part. You need to submit each part seperately on vocareum.

Problem 1 Regression (40 points)[¶](#Problem-1-Regression-(40-points))
----------------------------------------------------------------------

For this Assignment you are asked to implement linear regression. You are asked to implement 6 python functions for linear regression in linear\_regression.py. The input and output of the functions are specified in linear\_regression.py. Note that we have already appended the column of 1’s to the feature matrix, so that you do not need to modify the data yourself. We have provide a linear\_regression\_test.py for you to run some of your code and do some testing. Read the code, run it and check the result during your implementation.
 Submission: All you need to submit is linear\_regression.py

### Q1.1 Mean absolute Error (5 points)[¶](#Q1.1-Mean-absolute-Error-(5-points))

First, You need to implement the mean sqaure error. There should be simple as one or two lines of code and there is only one thing you need to be careful: please check the dimension of the input w, X and y before you do matrix dot operation. You might need to take a transpose of some matrix in order to get the right shape.
 Report the mean absolute error of the model on the give n test set, no rounding for the result. The type of the error is \<class 'numpy.float64'\>, you can do type(err) to check yours.

The Mean Absolute Error is given as

\$\$MAE = \\frac{1}{n} \\sum\_{i=1}\^n |y'-y|\$\$

-   (5 points) `TODO 1` You need to complete `def mean_absolute_error(w, X, y)` in `linear_regression.py`

### Q1.2 Linear Regression (5 points)[¶](#Q1.2-Linear-Regression-(5-points))

Now, you need to train your model. In linear regression model, it means find the weight for each feature. Check the lecture note page 21 and find the expression of general least absolute solution. Again, be careful with your matrix shape. Also, use numpy inverse funciton so you don't need to create your own. The shape of your return w should be (12,) if you train your model with the data we provide you
 Implement linear regression with no regularization and return the model parameters. Again, the implementation should be as simple as one or two lines of code. Use numpy instead of crazy nested for loop in your implementation. You don't need to worry about non-invertible matrix. We will take care of it in Q1.3.

-   (5 points) `TODO 2` You need to complete `def linear_regression_noreg(X, y)` in `linear_regression.py`

Once you finish Q1.1 and Q1.2, you should be able to run your linear\_regression\_test.py. Read the output, check your dimension of w, and MAE for training, evaluation and testing dataset. The MAE for all of them should between 0.5\~0.6. Otherewise, there must be something wrong with your implementation.

### Q1.3 Handle Non-invertible Matrix (5 points)[¶](#Q1.3-Handle-Non-invertible-Matrix-(5-points))

There are cases that during the calculation, the matrix are non-invertible. We manually created that situation in the data\_loader.py line 40. We simply manully set one row and one column in the dataset to be zero. Thus, we will get similar result as letcure note page 29. Now, you need to implement the solution on lecture note page 31. Here is the rul: in this assignment, if the smallest absolute value of all eigenvalue of a matrix is smaller than \$10\^{-5}\$, the matrix is non-invertible. If the matrix is non-invertible, keep adding \$10\^{-1}\*I\$ until it is invertible. You can use numpy functions to get eigen value for a matrix, get the absolute value ,find the minimum and create identity matrix, just search for them.

-   (10 points) `TODO 3` You need to complete `def linear_regression_invertible(X, y)` in `linear_regression.py`

Once you finish Q1.3, run linear\_regression\_test.py again, the MAE should be between 0.5\~0.6. Otherewise, there must be something wrong with your implementation.

### Q1.4 Regularized Linear Regression (5 points)[¶](#Q1.4-Regularized-Linear-Regression-(5-points))

To prevent overfitting, we usually add regularization. For now, we will focus on L2 regularization, same as the one in lecture note page 50. Implement the regularized\_linear\_regression(X,y,lambda)

-   (10 points) `TODO 4` You need to complete `def regularized_linear_regression(X, y, lambd)` in `linear_regression.py`

Once you finish Q1.4, run linear\_regression\_test.py again. Compare your result with Q1.3, did you find something interesting? Think about why or check the statement in red on lecture note page 50.

### Q1.5 Tune the Regularized Linear Hypter\_Parameter (10 points)[¶](#Q1.5-Tune-the-Regularized-Linear-Hypter_Parameter-(10-points))

Use your implementation from Q1.4, the regularized\_linear\_regression(X, y, lambd), try different lambds and find the best lambd to minimize the mae. Use Xval and yval to tune your hyper-parameter lambda
 tune\_lambda(Xtrain, ytrain, Xval, yval, lambds). To help us grading, set your lambd as all power of 10 from \$10\^{-19}\$ to \$10\^{19}\$ . However, in real world, no one knows the best range of hyper-parameters like lambda.

-   (10 points) `TODO 5` You need to complete `def tune_lambda(Xtrain, ytrain, Xval, yval)` in `linear_regression.py` Once you finish Q1.5, run linear\_regression\_test.py again. You should find a lambda that is \$1e\^{-14}\$. Otherewise, there must be something wrong with your implementation.

### Q1.6 Polynomial Regression (10 points)[¶](#Q1.6-Polynomial-Regression-(10-points))

We would like to introduce your another regression called polynomial regression. You can check the detail on wiki: [https://en.wikipedia.org/wiki/Polynomial\_regression](https://urldefense.proofpoint.com/v2/url?u=https-3A__en.wikipedia.org_wiki_Polynomial-5Fregression&d=DwMGaQ&c=clK7kQUTWtAVEOVIgvi0NU5BOUHhpN0H8p7CSfnc_gI&r=T3R02OWWnuhQEKisfuvCEA&m=2xIKowU1K7HfpqX0dPXhCPm9OeSNbznrqoGKtM8zPLI&s=QEunC4BCwl8JU7xZnw9lxtm5pYYzxf--ma_xt-bj5s8&e=) However, you don't need to check that to finish the homework
 Apply polinomial regression on the data, first try the second order function: \$y = w\_1\*X + w\_2\*X\^2\$ In this case, you should change the data \$[X]\$ to be \$[X X\^2]\$
 Implement the mapping\_data(x, power) function. For the higher order function, Eg.\$y = w\_1\*X + w\_2\*X\^2 + w\_3\*X\^3\$, thus your the data \$[X]\$ to be \$[X X\^2 X\^3...]\$. Be careful here, the new X is still 2D matrix, of size(N, D\*power). Use the insert function from numpy to quickly add data to original dataset. Obeserve how the mean absolute error change.

-   `TODO 6` You need to complete `def mapping_data(X, power)` in `linear_regression.py`

Once you finish Q1.6, you can run linear\_regression\_test.py, it will change the data with your mapping function, find corresponding w with , make prediction and report the mean absolute error again.
 You will see that your MAE keep increasing while the power goes up. Think about why this happen.

 This is end of part 1. Hopefully everything is clear and you didn't struggle too much. Now you should be able to take your next challenge, Part 2.

### Grading[¶](#Grading)

Your code will be graded on Vocareum with autograding script. For your reference, with my implementation the testing process is no longer than 15s including generating the grade report, and the processing time shows on the terminal is less than 0.2s. As long as you code can finish grading on Vocareum, you should be good.
 I can't provide any test case we used on the auto-grading script to you. However, I tried to help you indentify your problem with the printing statement. Please read them carefully.

