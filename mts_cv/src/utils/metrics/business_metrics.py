# -*- coding: utf-8 -*-
"""
    Metrics valuable for business
"""


def get_date_recognition_score(true_df, pred_df):
    """ Function returns percent of rightly recognized dates
    
    :param true_df: Dataframe of predicted values
    :param pred_df: Dataframe of true values
    :return:
    """
    all_df = pred_df.merge(true_df, on='names', suffixes=['_p', '_t'])
    score = 0
    for row in all_df.iterrows():
        date_p = row[1]['dates_p']
        date_t = row[1]['dates_t']
        
        if date_p == date_t:
            score += 1
    score = score / all_df.shape[0]
    return score
