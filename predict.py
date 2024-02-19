import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef

def classifiers(x_train, y_train, x_test,  y_test, model, args):

    if model == 'svm-rbf':
        model1 = SVC(probability=True)
    elif model == 'svm-linear':
        model1 = SVC(kernel= 'linear', probability=True)
    

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


    model1.fit(x_train, y_train)
    y_pred = model1.predict(x_test)
    y_proba = model1.predict_proba(x_test)[:, 1]

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

    print('{} Evaluation Report {}'.format(' -'*25,' -'*25))
    TN, FP, FN, TP = CM.ravel()
    print('True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}'.format(TN, FP,FN,TP))

    print('Accuracy: {0:.4f}%'.format(np.mean(accuray)))
    print('Sensitivity: {0:.2f}%'.format(((TP) / (TP + FN)) * 100.0))
    print('Specificity: {0:.2f}%'.format(((TN) / (TN + FP)) * 100.0))
    print('F1_Score: {0:.2f}'.format(np.mean(F1_Score)))
    print('MCC: {0:.2f}'.format(np.mean(MCC)))
    print('AUC: {0:.2f}'.format(np.mean(AUC)))

