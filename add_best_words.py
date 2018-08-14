from nltk import FreqDist, word_tokenize
def add_best_words(df, text_name, target_name, num_words, words_from="both"):
    '''
    Args:
        df: dataframe with text column and boolean target column
        text_name: name (string) of the text column
        target_name: name (string) of the target column
        num_words: the number of differentiating words to be returned
        words_from: "pos", "neg", or "both" - which text group you want to
            pull word candidates from, to be judged on impact as factors
    Returns:
        df: the df with best #(diff_count) words added: "best" here meaning words
        by which excerpts like text_1 can best be differentiated from excerpts
        like text_2 (e.g., spam vs ham), where best is defined as having maximum
        total frequency * freq-difference. Useful for generating the best words
        for spam filters, etc.
        keywords: the list of best words added to the df

    '''
    def score_word(df, target, factor):
        '''
        Args:
            df: dataframe with boolean target and factor columns
            target: name (string) of target data column (e.g., "isSpam)]
            factor: name (string) of factor to be scored for impact
        Returns:
            probabilistic impact of factor word (i.e., distance from even odds
            of 0.5) * number of occurrences
        '''
        total = len(df)
        positives = sum(df[target])

        # P(B|A)
        pos_occurrence_rate = sum(df[df[target]==True][factor]) / positives
        # P(A)
        pos_ratio = positives / total
        # P(B)
        total_occurrence_rate = sum(df[factor]==True) / total
        # P(B|A) * P(A) / P(B)
        conditional_prob = pos_occurrence_rate * pos_ratio / total_occurrence_rate

        return abs(.5 - conditional_prob) * total_occurrence_rate**.5

        # OLD METRIC
        # neg_occurrence_rate = sum(df[df[target]==False][factor]) / (len(df[target]) - sum(df[target]))
        # return (1 - min(pos_occurrence_rate, neg_occurrence_rate) / \
        #     max(pos_occurrence_rate, neg_occurrence_rate)) \
        #     * total_occurrence_rate

    # generate list of words to test
    num_to_test = max(num_words*4, num_words + 100)
    # NOTE: determining num_to_test is pretty arbitrary and can be toggled
    df[text_name] = df[text_name].apply(word_tokenize)
    pos_text = df[df[target_name]==True][text_name].sum()
    neg_text = df[df[target_name]==False][text_name].sum()
    fdist_pos = FreqDist(pos_text)
    fdist_neg = FreqDist(neg_text)
    pos_most_common = [tup[0] for tup in fdist_pos.most_common(num_to_test) if tup[0] not in [".", "do"]]
    neg_most_common = [tup[0] for tup in fdist_neg.most_common(num_to_test) if tup[0] not in [".", "do"]]
    if words_from == "pos":
        word_list = pos_most_common
    elif words_from == "neg":
        word_list = neg_most_common
    elif words_from =="both":
        # this looks odd b/c one might be shorter than num_to_test, and it's unclear which
        # first, remove duplicates
        pos_most_common = [word for word in pos_most_common if word not in neg_most_common]
        word_list = pos_most_common[:min(num_to_test//2, len(pos_most_common)-1)]
        word_list += neg_most_common[:min(num_to_test-len(word_list), len(neg_most_common)-1)]
    else:
        raise ValueError("invalid words_from input: must be \"pos\", \"neg\", or \"both\"")
    if len(word_list) < num_to_test/2:
        raise ValueError("Texts are too small for the diff_count you submitted.  Try a smaller num_words value.")

    # add column to df for each word, record word scores
    word_scores = {}
    for word in word_list:
        df[word] = [word in li for li in df[text_name]]
        word_scores[word] = score_word(df, target_name, word)

    # remove word-columns whose scores are outside the top #num_words
    cutoff = sorted(word_scores.values())[-num_words]
    cut_words = [word for word in word_scores.keys() if word_scores[word] < cutoff]
    for word in cut_words:
        del df[word]

    best_words = [word for word in word_scores.keys() if word_scores[word] >= cutoff]
    return df, best_words


# DRIVER
# import numpy as np
# import pandas as pd
# import scipy
# import sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#
# filepath = os.path.join(THIS_FOLDER, "sentiment_data/yelp_labelled.txt")
# yelp_df = pd.read_csv(filepath, delimiter="\t", header=None)
# yelp_df.columns = ["review", "sentiment"]
#
# yelp_df = add_best_words(yelp_df, "review", "sentiment", 10)
