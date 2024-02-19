import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, \
        confusion_matrix, auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef

def classifiers(x_train, y_train, x_test,  y_test, train_seq, test_seq, model, args):

    accuray = []
    F1_Score = []
    AUC = []
    MCC = []
    Recall = []

    Results = []
    CM = np.array([
        [0, 0],
        [0, 0],
    ], dtype=int)

    # Model selection
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

    accuray.append(accuracy_score(y_test, y_pred))
    F1_Score.append(f1_score(y_test, y_pred))
    MCC.append(matthews_corrcoef(y_test, y_pred))
    Recall.append(recall_score(y_test, y_pred))

    FPR, TPR, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(FPR, TPR)
    AUC.append(roc_auc)

    accuray = [_ * 100.0 for _ in accuray]
    Results.append(accuray)

    # Writing prediction in csv file
    true_val = ["Positive" if label == 1 else "Negative" for label in y_test]
    pred_val = ["Positive" if pred == 1 else "Negative" for pred in y_pred]

    df = pd.DataFrame({'Sequences': test_seq, 'True Value': true_val, 'Prediction': pred_val})
    df.to_csv('output.csv',index=False)

    # Performance metrics
    print('{} Evaluation Report {}'.format(' -'*25,' -'*25))
    TN, FP, FN, TP = CM.ravel()
    print('True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}'.format(TN, FP,FN,TP))

    print('Accuracy: {0:.4f}%'.format(np.mean(accuray)))
    print('Sensitivity: {0:.2f}%'.format(((TP) / (TP + FN)) * 100.0))
    print('Specificity: {0:.2f}%'.format(((TN) / (TN + FP)) * 100.0))
    print('F1_Score: {0:.2f}'.format(np.mean(F1_Score)))
    print('MCC: {0:.2f}'.format(np.mean(MCC)))
    print('AUC: {0:.2f}'.format(np.mean(AUC)))

