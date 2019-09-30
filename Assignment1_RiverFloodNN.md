

```python
import numpy as np
```

### Calculate z1
Because the hidden layer will have 4 perceptrons, we need 4x2 weight matrix (W1) and 4x1 bias matrix (b1) using numpy

    a1        = sigmoid ( W1(T)   a0 + b1)

    4 x 1     = sigmoid ( [4x2 ]  [ 2x1] + [4x1] )


```python
# W1 is  4x2
np.random.seed(seed=123)
W1= np.random.rand(4,2)
b1 = np.random.rand(4,1)
x = np.array([0.4, 0.32])
```


```python
print("W1 = " , W1, "\n")
print("b1 = " , b1)
```

    W1 =  [[0.69646919 0.28613933]
     [0.22685145 0.55131477]
     [0.71946897 0.42310646]
     [0.9807642  0.68482974]] 
    
    b1 =  [[0.4809319 ]
     [0.39211752]
     [0.34317802]
     [0.72904971]]
    

#### z1 = W1.dot(x) + b1


```python
# multiply the matrices
m1 = np.asmatrix(W1.dot(x)) 
m1
```




    matrix([[0.37015226, 0.26716131, 0.42318166, 0.6114512 ]])




```python
z1 = np.empty((4,1), dtype = object)
```


```python
# add the matrices

for i in range(len(m1)):
   for j in range(len(m1[0])):
       z1[i][j] = m1[i][j] + b1[i][j]
```


```python
z1 = z1[0][0]
z1
```




    matrix([[0.85108416, 0.74809321, 0.90411356, 1.0923831 ]])




```python
a1 = 1/(1+np.exp(-z1))
print(a1)
```

    [[0.70079452 0.67876308 0.71179411 0.74883021]]
    

### Calculate z2
Because the output layer has 1 perceptrons, we need 4x2 weight matrix (W2) and 1x1 bias matrix (b2)

    a2        = sigmoid ( W2(T)   a1 + b2)

    1 x 1     = sigmoid ( [4x1 ]  [ 1x4] + [1x1] )


```python
np.random.seed(seed=123)
W2 = np.random.rand(4,1)
b2 = np.random.rand(1,1)

```


```python
print("W2 = " , W2, "\n")
print("b2 = " , b2)
```

    W2 =  [[0.69646919]
     [0.28613933]
     [0.22685145]
     [0.55131477]] 
    
    b2 =  [[0.71946897]]
    

#### z2 = W2.dot(z1) + b2



```python
# multiply the matrices
m2 = np.asmatrix(W2.dot(z1)) 
m2
```




    matrix([[0.59275389, 0.52102387, 0.62968723, 0.76081117],
            [0.24352866, 0.21405889, 0.25870245, 0.31257377],
            [0.19306968, 0.16970603, 0.20509947, 0.24780869],
            [0.46921527, 0.41243483, 0.49845116, 0.60224693]])




```python
z2 = np.empty((1,1), dtype = object)
```


```python
# add the matrices
for i in range(len(m2)):
   for j in range(len(m2[0])):
       z2[i][j] = m2[i][j] + b2[i][j]
```


```python
z2 = 
a2 = 1/(1+np.exp(-z2)) # y
```


```python
# what is your conclusion? 
```
