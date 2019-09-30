
#### River Flood Problem (solved with NN)

Two sensors are installed in Susquehanna river that can measure the level of the water on two different locations. Their measurements goes through a neural network and the output shows if there is a chance of flood or not (y=1 means there is a chance of flood). Suppose sensor 1 shows 0.4 feet while the second one shows 0.32 feet. Is there any chance of flood? 

1. Please use Python to answer to the question.  

2. Please generate Random vectors and matrices (using numpy) for weights and bias. (seed =123)  

3. You must explain your final result.

#### Submitted by : Deepti Gupta


```python
import numpy as np
```

### Calculate z1
Because the hidden layer will have 4 perceptrons, we need 4x2 weight matrix (W1) and 4x1 bias matrix (b1) using numpy

    a1        = sigmoid ( W1(T) .  x + b1)

    4 x 1     = sigmoid ( [4x2 ]  [ 2x1] + [4x1] )


```python
# W1 is  4x2
np.random.seed(seed=123)
W1= np.random.rand(4,2)
b1 = np.random.rand(4,1)
x = np.matrix('0.4; 0.32')
```


```python
print("W1 = " , W1, "\n")
print("b1 = " , b1, "\n")
print("x = " , x)
```

    W1 =  [[0.69646919 0.28613933]
     [0.22685145 0.55131477]
     [0.71946897 0.42310646]
     [0.9807642  0.68482974]] 
    
    b1 =  [[0.4809319 ]
     [0.39211752]
     [0.34317802]
     [0.72904971]] 
    
    x =  [[0.4 ]
     [0.32]]
    

#### z1 = W1.dot(x) + b1


```python
z1 = W1.dot(x) + b1
z1
```




    matrix([[0.85108416],
            [0.65927883],
            [0.76635967],
            [1.3405009 ]])




```python
a1 = 1/(1+np.exp(-z1))
print(a1)
```

    [[0.70079452]
     [0.65909837]
     [0.68273289]
     [0.7925723 ]]
    

### Calculate z2
Because the output layer has 1 perceptrons, we need 4x2 weight matrix (W2) and 1x1 bias matrix (b2)

    a2        = sigmoid ( W2(T)   a1 + b2)

    1 x 1     = sigmoid ( [1x4 ]  [ 4x1] + [1x1] )


```python
np.random.seed(seed=123)
W2 = np.random.rand(1,4)
b2 = np.random.rand(1,1)

```


```python
print("W2 = " , W2, "\n")
print("b2 = " , b2, "\n")
print("a1 = " , a1)
```

    W2 =  [[0.69646919 0.28613933 0.22685145 0.55131477]] 
    
    b2 =  [[0.71946897]] 
    
    a1 =  [[0.70079452]
     [0.65909837]
     [0.68273289]
     [0.7925723 ]]
    

#### z2 = W2.dot(z1) + b2



```python
z2 = W2.dot(a1) + b2
z2
```




    matrix([[1.98798049]])




```python
a2 = 1/(1+np.exp(-z2[0][0]))
a2
```




    matrix([[0.87952932]])



#### Conclusion :  When sensor 1 shows 0.4 feet while the second one shows 0.32 feet, there is a greater chance of flooding with 87.95% probability 


```python

```
