import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings

from time import time

def cross_test_model(model, X_set, y):
    start = time()
    scores = cross_validate(model, X_set, y, cv=3,
                            scoring=['roc_auc', 'f1'], return_train_score=True)
    print("  Mean train AUROC:", round(np.mean(scores['train_roc_auc']), 4))
    print("  Mean test AUROC:", round(np.mean(scores['test_roc_auc']), 4))
    print("  Mean train F1-score:", round(np.mean(scores['train_f1']), 4))
    print("  Mean test F1-score:", round(np.mean(scores['test_f1']), 4))
    print("Mean fit/score time:", round( (np.mean(scores['fit_time']) + np.mean(scores['score_time']))/60, 1),
                                       'minutes')
def quick_test_model(model, X_set, y):
    start = time()
    X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=1/3)
    model.fit(X_train, y_train)

    y_train_proba = model.predict_proba(X_train)[:,1]
    y_test_proba = model.predict_proba(X_test)[:,1]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_auroc = round(roc_auc_score(y_train, y_train_proba), 3)
    test_auroc = round(roc_auc_score(y_test, y_test_proba), 3)
    train_f1 = round(f1_score(y_train, y_pred_train), 3)
    test_f1 = round(f1_score(y_test, y_pred_test), 3)

    print("  train AUROC:", train_auroc)
    print("  test AUROC:", test_auroc)
    print("  train f1-score:", train_f1)
    print("  test f1-score:", test_f1)
    print("elapsed time:", round((time()-start)/60, 1), 'minutes')

def test_models(model_tups, X_set, y):
    ''' model_tups = [ (model, name), (model, name) ]'''

    warnings.filterwarnings(action='ignore')
    # UndefinedMetricWarning arises with f1-scores with 0 precision or recall.
    # In that case the function sets the score to 0 and returns a divide-by-zero
    # warning.  I only need to know that the F1 is zero - if I want a better look
    # at the errors, I'll look at the confusion matrix

    X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=1/3)

    for tup in model_tups:
        start = time()
        model = tup[0]
        model.fit(X_train, y_train)
        y_train_proba = model.predict_proba(X_train)[:,1]
        y_test_proba = model.predict_proba(X_test)[:,1]
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_auroc = round(roc_auc_score(y_train, y_train_proba), 3)
        test_auroc = round(roc_auc_score(y_test, y_test_proba), 3)
        train_f1 = round(f1_score(y_train, y_pred_train), 3)
        test_f1 = round(f1_score(y_test, y_pred_test), 3)

        print(tup[1] + ':')
        print("  train AUROC:", train_auroc)
        print("  test AUROC:", test_auroc)
        print("  train f1-score:", train_f1)
        print("  test f1-score:", test_f1)
        print("elapsed time:", round((time()-start)/60, 1), 'minutes\n')

def show_conf_mat(model, X_set, y):
    X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=1/3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.figure(figsize=(5,5))
    conf_mat = np.round(confusion_matrix(y_test, y_pred)/len(y_test), 2)
    conf_mat = pd.DataFrame(np.fliplr(np.rot90(conf_mat)), # because sklearn arranges it counter to convention
             columns=['actual_True', 'actual_False'],
             index=['predicted_True', 'predicted_False'])
    sns.heatmap(conf_mat, square=True, annot=True, annot_kws={'size':16}, cmap='RdBu_r', center=0, fmt='g')
    plt.title('confusion matrix (shows percentage of \n full dataset contained in each quadrant)')
    plt.tight_layout()
    plt.show()
