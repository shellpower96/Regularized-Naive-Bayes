# Regularized-Naive-Bayes
The proposed 'regularized naive Bayes' combines 'class-dependent attribute weights' and 'class-independent attributes weights' together to derive the regularized posterior function, and they are optimized simultanuouely by using L-BFGS algorithm.
![](https://github.com/Shellson/Regularized-Naive-Bayes/blob/main/frame4.png)
## Dataset
The example dataset is iris.csv in which the first four columns are features and the last one is class label. The detailed descriptions of all datasets used in the paper can be found in https://archive.ics.uci.edu/ml/index.php.

All the features should be first convert to numerical value, and then make a classification by RNB.

If you use this code, please cite:
<br>
  @article{wang2020regularized,
    title={A regularized attribute weighting framework for naive bayes},
    author={Wang, Shihe and Ren, Jianfeng and Bai, Ruibin},
    journal={IEEE Access},
    volume={8},
    pages={225639--225649},
    year={2020},
    publisher={IEEE}
  }
<br>
