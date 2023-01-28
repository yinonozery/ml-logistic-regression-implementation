# Machine Learning - Logistic Regression Implementations
This project contains two implementations of logistic regression using the one-vs-rest strategy for multi-class classification: one that I've written from scratch, and another one using the scikit-learn library. The models are trained and tested on random data samples with two features using.
The goal of this project is to compare the performance and accuracy of both implementations and to provide a clear understanding of how linear regression works.

## Input
```
<Number of random samples> : <int>
<Number of classes 2/3/4> : <int>
```

## Output
```
Confusion Matrix
Accuracy
Precision
Recall
F1 Score
```
___
## Example
#### Input
```
<Number of random samples> : 1000
<Number of classes 2/3/4> : 3
```

#### Output
##### * Scikit-learn implementation

![image](https://user-images.githubusercontent.com/74764366/215270025-08e4dd4e-fbb5-45e4-b932-fd91d7981249.png)

![image](https://user-images.githubusercontent.com/74764366/215270030-4c516b00-9235-49e4-b611-ea97a91b3331.png)


##### * Manual implementation

![image](https://user-images.githubusercontent.com/74764366/215270091-1d15d1c9-f30d-4723-a0ea-3d5ec678e133.png)

![image](https://user-images.githubusercontent.com/74764366/215270094-e61076f3-8509-4685-8ed6-1d5841473e28.png)
