# Machine Learning on Various Datasets 
Alec Arreche

## Car Evaluation 

| Model                  | Accuracy | Training Time | Model Size     | Testing Time |
| -----------------------| ---------| ------------- | -------------- | ------------ |
| Naive Bayes Classifier | 0.871    | 0.0086s       | 1.266 KiB      | 0.0032s      |
| Decision Tree          | 0.948    | 0.4208s       | 59.460 MiB     | 0.000594s    |
| Support Vector Machine | 0.998    | 1.667s        | 13.379 MiB     | 0.0067s      |
| Multilayer Perceptron  | 0.987    | 22.960s       | 16.206 MiB     | *            |

Training and testing datasets were separated using SkLearn's `train_test_split` function, with testing size of 30%. 
Since all features are categorical, and there were not too many, we applied One Hot Encoding to create numerical features. 
Additionally, we Label Encoded the various classes of the response. 
The Naive Bayes Classifier can be looked at as a baseline since it is a simpler model with no hyperparameter tuning. 
For the decision tree, we use cross validation to find the optimal tree depth by testing all depths between 1 and 30. 
We then applied this best model on a separate training dataset. 
To tune the SVM, we performed cross validated grid search on the kernel and `C` hyperparameters using the built in functionality
in Sci-kit Learn. 
Lastly, for MLP, we perfom a similar grid search tuning method on activation functions and hidden layer sizes, but we do not perform cross validation since training time would take too long. 
Since testing for this model was part of the tuning process, testing time was not included. 
Ultimately, these covariates are very predictive of the outcome, with the SVM offering superior accuracy at little cost of training time and memory. 

## Abolone 

| Model                  | Accuracy | Training Time | Model Size     | Testing Time |
| -----------------------| ---------| ------------- | -------------- | -------------|
| Naive Bayes Classifier | 0.572    | 0.00462s      | 1.250 KiB      | 0.00215s     |
| Decision Tree          | 0.614    | 1.231s        | 113.211 MiB    | 0.00054s     |
| Support Vector Machine | 0.656    | 21.975s       | 28.025 MiB     | 0.209s       |
| Multilayer Perceptron  | 0.656    | 33.431s       | 265.531 KiB    | *            |

Training and testing datasets were separated using SkLearn's `train_test_split` function, with testing size of 30%. 
Since `rings` had levels ranging 1 to 28, we binned them into three classes similar to one of the linked papers on the data. 
These classes are constructed to minimize class imbalance and create three classes of age of abalone. 
The categorical feature, `sex` was then encoded using One Hot Encoding. 
Similarly, no hyperparameter turning was performed on the Naive Bayes Classifier. 
The Decision Tree depth was tuned using cross validation on all depths between 1 and 30 and then validated on a separate training dataset. 
The SVM was tuned with cross validation with cross validated grid search on the kernel and `C` using the built in functionality. 
The MLP layer sizes and activation function was also performed with grid search, but since training time is so long, we do not perform cross validation. 
Since testing was part of the tuning process, the testing time is included in the training time. 
The MLP performs the best in this case, and can probably be tuned to perform better but at the cost of largely increased training time. 

## Madelon 

| Model                  | Accuracy | Training Time | Model Size     | Testing Time |
| -----------------------| ---------| ------------- | -------------- | -------------|
| Naive Bayes Classifier | 0.592    | 0.00655s      | 1.250 KiB      | 0.0024s      |
| Decision Tree          | 0.795    | 37.965s       | 1.166 GiB      | 0.00153s     |
| Support Vector Machine | 0.698    | 3.862s        | 59.328 KiB     | *            |
| Multilayer Perceptron  | 0.602    | 194.241s      | 189.594 KiB    | *            |

Training and testing datasets were separated using SkLearn's `train_test_split` function, with testing size of 30%. 
This dataset was unique because it has 20 "true" features and 480 noise features.
However, since the goal of this project is not to perform feature extraction, we attempt to fit on all features. 
The Naive Bayes Classifier, all without any tuning, set a baseline of around 0.6 testing accuracy on all 500 features. 
Since the class distribution of the data is 50/50 by construction, we see that model is able to somewhat predict the outcome. 
As an experiement we take a 100 samples of 100 features on fit the model on those chosen features. 
This creates a 99% probabiliy that at least one feature will be one of the true features. 
The Naive Bayes Classifier was able to get a testing accuracy of roughly over 0.6 on one of these samples, but not significantly better than the model on all 500 features. 
The tree model was created using cross validation on depth and was able to perform with a testing accuracy of almost 0.8. 
Due to large training time, the SVM was tuned only on C with no cross validation. 
The kernel was set to radial due to the large nonlinearity in the relationship, and the model was able to perform with almost 0.7 testing accuracy. 
THe MLP was tuned on layer sizes and alpha, without cross validation due to training time. 
Alpha was chosen because the regularization parameter penalizes many features in model, so it was used with hope to perform implicit feature selection. 
Since testing was used to tune the MLP and SVM, the testing time is part of the training time. 
The tree model here performs best due to its nature of creating implicit feature extraction at its decision trees, while the other models are unable to offer the same performance. 

## KDD

| Model                  | Accuracy | Training Time | Model Size     | Testing Time |
| -----------------------| ---------| ------------- | -------------- | -------------|
| Naive Bayes (binary)   | 0.980    | 35.733s       | 1.227 GiB      | 9.197s       |
| Naive Bayes (labeled)  | 0.951    | 32.852s       | 1.226 GiB      | 12.472s      |
| Decision Tree          | 0.999    | 297.775s      | 10.516 GiB     | *            |
| Support Vector Machine | 0.994    | 443.809s      | 8.763 GiB      | *            |
| Multilayer Perceptron  | 0.996    | 65.383 s      | 1.227 GiB      | 9.313s       |

Training and testing datasets were separated using SkLearn's `train_test_split` function, with testing size of 30%. 
This data had many classes with huge class imbalance. To remedy this, we aggregated the classes into "normal" and "attack". 
This still had a minor class imbalance of 80% normal and 20% attack, but prior to creating binary classes, some classes only had 1 observation in millions of examples. 
The categorical features were then encoded using One Hot Encoding. 
We attempted to fit a Naive Bayes Classifier on the non binary clases, and we see impressive results of 95% accuracy. 
This continues with the model on binary class data which had accuracy of almost 98%.
Both were fit without any hyperparameter tuning. 
The tree model's depth and the SVM's C were tuned by fitting the model on training data and testing on the other split, but without cross validation to limit time. 
Since the tree and SVM were tested as part of the tuning process, the testing time is part of the training time. 
The MLP was unable to be tuned due to its long training time on such a large dataset. 

# Conclusion

Throughout this project we observe how various models performed on different datasets with increasing number of features and observations. 
The smaller datasets were able to be thoroughly tuned using cross validation and grid search in a reasonable amount of time. 
With a large number of features of observations however, there exists a tradeoff where the extent of the tuning had to be sacrificed in able to get a model trained in a short enough period of time. 
Additionally, we observe how different models perform under differnt types of data. 
For example, when the number of significant features is sparse, as was the case with the abalone data, the tree models were able to out perform the other models with its implicit feature selection. 
With other data, such as the KDD dataset, the simple Naive Bayes Classifier was able to perform with nearly identical accuracy at comparatively instant training time.
Lastly, we observe how the neural network was able to provide comparable results to the other, thoroughly tuned models with little tuning. 
Therefore, with additional computation resources, this model could be more thoroughly tuned and potentially outperform these results. 