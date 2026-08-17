# Credit Card Fraud Detection

A machine learning project comparing a manually implemented **Perceptron** with an **Artificial Neural Network (ANN)** for detecting fraudulent credit card transactions.

The project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing **284,807 transactions**. Fraud represents only about **0.172%** of the data, making this a highly imbalanced classification problem.

## Models

### Perceptron

A simple linear classifier used as a baseline.

Random oversampling was applied to the training data so that fraud examples appeared more often during training.

### Artificial Neural Network

Implemented using TensorFlow/Keras with:

* Hidden layers: 64, 32 and 16 neurons
* Batch Normalization
* ReLU activation
* He Normal initialization
* AdamW optimizer
* Binary Focal Crossentropy
* Validation-based threshold tuning

## Evaluation

The models were evaluated using:

* Precision
* Recall
* F1-score
* False Positive Rate
* PR-AUC
* Confusion Matrix

These metrics are more useful than accuracy alone because fraud transactions are extremely rare.

## Results

| Metric              | Perceptron |         ANN |
| ------------------- | ---------: | ----------: |
| Accuracy            |     87.49% |  **99.93%** |
| Precision           |      1.26% |  **75.93%** |
| Recall              | **92.86%** |      83.67% |
| F1-score            |      2.49% |  **79.61%** |
| False Positive Rate |     12.52% |   **0.05%** |
| PR-AUC              |        N/A | **0.75121** |
| Fraud detected      |         91 |          82 |
| False alarms        |      7,120 |      **26** |

The Perceptron detected more fraud cases but produced **7,120 false alarms**. The ANN reduced this to only **26 false alarms** while still detecting 82 of the 98 fraud transactions.

## Results Visualization

![Threshold vs Precision, Recall and F1-score](results/threshold_metrics.png)

![Precision-Recall Curve](results/precision_recall_curve.png)

## Technologies

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

## Conclusion

The ANN performed better overall because it provided a much better balance between detecting fraud and avoiding false alerts.

Its final performance was:

```text
Precision: 75.93%
Recall:    83.67%
F1-score:  79.61%
PR-AUC:    0.75121
```

## Author

**Hristijan Kochovski**
UP FAMNIT
