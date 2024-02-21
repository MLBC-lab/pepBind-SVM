import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, \
        confusion_matrix, auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef

def classifiers(x_train, y_train, x_test,  y_test, train_seq, test_seq, model, args):
    """
    Train a classifier based on the specified model and evaluate its performance on a test set.
    
    Args:
        x_train (numpy array): Training features.
        y_train (numpy array): Training labels.
        x_test (numpy array): Testing features.
        y_test (numpy array): Testing labels.
        train_seq (Series): Training sequences (not used in training, included for potential future use).
        test_seq (Series): Testing sequences for output file.
        model (str): Model type ('svm-rbf' or 'svm-linear').
        args (Namespace): Additional arguments, not used in this function but included for future extensions.
    
    Outputs:
        Writes the model predictions and true values to 'output.csv'.
        Prints the evaluation report including various performance metrics.
    """
    # Initialize lists to store performance metrics
    accuracy, F1_score, AUC, MCC, recall = 0, 0, 0, 0, 0
    CM = np.zeros((2, 2), dtype=int)

    # Model selection based on the specified model type
    if model == 'svm-rbf':
        model1 = SVC(probability=True)
    elif model == 'svm-linear':
        model1 = SVC(kernel= 'linear', probability=True)
    
    # Training and inference
    model1.fit(x_train, y_train)
    y_pred = model1.predict(x_test)
    y_proba = model1.predict_proba(x_test)[:, 1]

    # Calculating performance metrics
    CM += confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred) * 100
    F1_score = f1_score(y_test, y_pred)
    MCC = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    FPR, TPR, _ = roc_curve(y_test, y_proba)
    AUC = auc(FPR, TPR)

    # Writing prediction to a csv file
    true_val = ["Positive" if label == 1 else "Negative" for label in y_test]
    pred_val = ["Positive" if pred == 1 else "Negative" for pred in y_pred]

    df = pd.DataFrame({'Sequences': test_seq, 'True Value': true_val, 'Prediction': pred_val})
    df.to_csv('output.csv',index=False)

    # Print performance metrics
    print('{} Evaluation Report {}'.format(' -'*25,' -'*25))
    TN, FP, FN, TP = CM.ravel()
    print('True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}'.format(TN, FP,FN,TP))
    print('Accuracy: {0:.2f}%'.format(accuracy))
    print('Sensitivity: {0:.2f}%'.format(((TP) / (TP + FN)) * 100.0))
    print('Specificity: {0:.2f}%'.format(((TN) / (TN + FP)) * 100.0))
    print('F1_Score: {0:.2f}'.format(F1_score))
    print('MCC: {0:.2f}'.format(MCC))
    print('AUC: {0:.2f}'.format(AUC))

