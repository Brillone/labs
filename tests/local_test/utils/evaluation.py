import numpy as np
import json


def mse(true_vals, prediction_vals):
    return np.mean((prediction_vals - true_vals)**2)


def mae(true_vals, prediction_vals):
    return np.mean(np.abs(prediction_vals - true_vals))


metrics_callables = {'MSE': mse,
                     'MAE': mae}


def evaluate(true_vals, prediction_vals, metrics, data_set_name=''):
    scores = {}

    for metric in metrics:
        metric_callable = metrics_callables.get(metric)
        metric_score_dict = {data_set_name + metric: metric_callable(true_vals, prediction_vals)}
        scores.update(metric_score_dict)

    return scores


def summarize_scores(raw_scores):
    scores = {}

    for scores_key in raw_scores.keys():
        scores[scores_key] = np.mean(raw_scores[scores_key])

    return scores


def save_model_scores(scores, save_path):
    with open(save_path + '/' + 'scores' + '.json', 'w') as f:
        json.dump(scores, f, indent=4)