import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss

def top_predictions(X, y, model, n_predictions, test_size=1/3, verbose=True):
    '''
    Args:
        X: array-like with shape [num_examples, num_features]
        y: array-like with shape [num_examples,]
        model: any predictive model with a predict_proba() function with a return
               array formatted like sklearn's predict_proba() return array
        n_predictions: int < nunique(y), gives number of top predictions desired
        test_size: float between 0 and 1, gives size of test set for train/test split
        verbose: bool, prints report on results

    Returns:
        predictions_df: pandas dataframe with shape (num_examples, n_predictions*2)
                        and columns 'prediction_1', 'prediction_2'... 'probability_1',
                        'probability_2'... etc.
                        In cases where prediction_1 has a predict_proba value of 1.0,
                        prediction_2... prediction_n are None
    '''

    # NOTE on the encoder: this function keeps y_test un-encoded because it may
    # contain classes not in y_train, rendering the predict_proba output indecipherable,
    # since its columns are 0-->len(unique(y_train)) on the assumption that y_test
    # is restricted to vals in y_train)

    if n_predictions > len(np.unique(y)):
        return ValueError("Number of top predictions requested cannot exceed number of classes")

    start = time()
    if verbose:
        print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Parse tiered probabilities
    probs = pd.DataFrame(model.predict_proba(X_test))
    max_column_indices = np.argsort(probs.values, axis=1)[:, -n_predictions:]
    sorted_probs = np.sort(probs.values, axis=1)[:, -n_predictions:]
    col_columns = ["prediction_" + str(i) for i in range(n_predictions, 0, -1)]
    columns_df = pd.DataFrame(encoder.inverse_transform(max_column_indices), columns=col_columns)
    val_columns = ["probability_" + str(i) for i in range(n_predictions, 0, -1)]
    probs_df = pd.DataFrame(sorted_probs, columns=val_columns)
    predictions_df = pd.concat([columns_df, probs_df], axis=1)

    # sort columns to prediction_1, prediction_2... probability_1, probability_2...
    prediction_indices = [i for i in range(n_predictions-1, -1, -1)]
    probability_indices = [i + n_predictions for i in prediction_indices]
    predictions_df = predictions_df.iloc[:, prediction_indices + probability_indices]

    # replace zero-probability predictions with None
    is_absolute = pd.Series(predictions_df.probability_1 >= 0.99)
    prediction_cols = predictions_df.columns[:n_predictions]
    for col in prediction_cols[1:]:
        predictions_df.loc[is_absolute, col] = None

    if verbose:
        # calculate and print model accuracy & runtime
        train_accuracy = round(sum(y_pred_train==y_train)/len(y_train), 3)
        test_accuracy = round(sum(encoder.inverse_transform(y_pred_test)==y_test)/len(y_test), 3)
        print("  train accuracy:", train_accuracy)
        print("  test accuracy:", test_accuracy)
        print("elapsed time:", round((time()-start)/60, 1), 'minutes')

        # calculate stats for the cumulative n_predictions output
        pct_absolute = round(100*sum(predictions_df.probability_1 >= 0.99)/predictions_df.shape[0], 1)

        prediction_cols = predictions_df.columns[:n_predictions]
        probability_cols = predictions_df.columns[n_predictions:]

        # average probability of prediction for non_absolute predictions
        av_non_abs_probabilities = []
        for i in range(n_predictions):
            av_non_abs_probabilities.append(
                round( predictions_df[ predictions_df.probability_1 < 0.99 ][probability_cols[i]].mean(), 2)
            )

        # absolute vs. non-absolute prediction accuracies
        y_test = y_test.reset_index(drop=True)
        is_absol = pd.Series(predictions_df.probability_1 >= 0.99)
        absol_accuracy = round(sum(predictions_df.prediction_1[is_absol]==y_test[is_absol])/sum(is_absol), 3)
        isnt_absol = pd.Series(predictions_df.probability_1 < 0.99)
        unabsol_accuracy = round(sum(predictions_df.prediction_1[isnt_absol]==y_test[isnt_absol])/sum(isnt_absol), 3)

        # cumulative accuracy of top n_predictions (y_true is one of n_predictions) for non-absolute predictions
        total_accuracy_non_absol = 0
        for i in range(n_predictions):
            total_accuracy_non_absol += \
                sum(predictions_df[prediction_cols[i]][isnt_absol]==y_test[isnt_absol])/sum(isnt_absol)
        total_accuracy_non_absol = round(total_accuracy_non_absol, 3)

        # cumulative accuracy of top n_predictions (y_true is one of n_predictions) for all predictions
        total_accuracy = 0
        for i in range(n_predictions):
            total_accuracy += \
                sum(predictions_df[prediction_cols[i]]==y_test)/len(y_test)
        total_accuracy = round(total_accuracy, 3)

        # log loss
        y_true = (predictions_df.prediction_1==y_test)
        y_pred = predictions_df.probability_1
        primary_log_loss = round(log_loss(y_true, y_pred), 3)
        primary_brier_score = round(brier_score_loss(y_true, y_pred), 3)

        # brier score
        y_true = sum(
            [ (predictions_df[prediction_cols[i]] == y_test).astype(int) \
                for i in range(n_predictions) ]
        )
        y_pred = (predictions_df.probability_1 + predictions_df.probability_2 + predictions_df.probability_3)
        total_log_loss = round(log_loss(y_true, y_pred), 3)
        total_brier_score = round(brier_score_loss(y_true, y_pred), 3)

        # print results
        print("\nPercent of predictions that are absolute (probability > 0.99):   {}%".format(pct_absolute))
        print("\nAccuracies")
        print("  Accuracy of absolute predictions:                             ", absol_accuracy)
        print("  Accuracy of primary non-absolute predictions:                 ", unabsol_accuracy)
        print("  Cumulative accuracy of all top predictions when non-absolute: ", total_accuracy_non_absol)
        print("  Total cumulative accuracy of all top predictions in all cases:", total_accuracy)

        print("\nWhen prediction is not absolute, average probability value returned for: ")
        for i in range(1, n_predictions+1):
            print("  prediction {}: {}".format(i, av_non_abs_probabilities[i-1]))

        print("\nLog-loss and brier scores:")
        print("  Primary prediction log loss:           ", primary_log_loss)
        print("  Primary prediction brier score:        ", primary_brier_score)
        print("  Cumulative top predictions log loss:   ", total_log_loss)
        print("  Cumulative top predictions brier score:", total_brier_score)

    return predictions_df
