# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
### Program to implement the the Logistic Regression Using Gradient Descent.
```py
# Developed by: SHALINI.K
# RegisterNumber:  212222240095
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)

```

## Output:
![1](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/4c68e40e-3945-4836-8d82-06c65dd83ef5)

![2](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/681774f8-6006-4015-ab83-4d0ef73b56b9)

![3](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/95f47b4e-8f31-4b66-acef-8c52928a190d)

![4](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/a146f0fd-7ae3-421f-aed3-843b4b6bccd0)

![5](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/ca1e5830-e845-4ceb-92d1-48c9584d3d99)

![6](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/9e211658-4dad-48af-99f8-a855eb5d4ad1)

![7](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/e4bc03b8-1ac6-4797-87c7-74ea6579db22)

![8](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/ae9dfe14-cd80-4953-9f5e-074d9b85ecf9)

![9](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/6e2cdfec-28fb-4bca-b76f-a07908d9145d)

![10](https://github.com/shalinikannan23/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118656529/ccf62739-379f-4b62-8634-98ee89693767)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

