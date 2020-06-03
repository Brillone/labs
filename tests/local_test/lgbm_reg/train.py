# base
from labs import experiment_path
import json
import argparse

# internal
from tests.local_test.lgbm_reg.modeling import create_model, save_model
from tests.local_test.utils.data_handling import load_data, create_cv_data
from tests.local_test.utils.evaluation import evaluate, save_model_scores, summarize_scores
from tests.local_test.lgbm_reg.config import data_config, cv_config


@experiment_path
def lgbm_reg(experiment, artifacts_path, metrics):
    # init
    data = load_data(data_config['data_path'])
    cv_data = create_cv_data(data['X_train'], data['y_train'], cv_config=cv_config)

    # hyperparams
    model = create_model(experiment['hyperparams'])

    # data config
    features_names = data_config['features_names']
    target_name = data_config['target_name']

    # scores dict
    scores = {'raw_cv_scores': {}, 'cv_scores': {}, 'test_scores': {}}

    # cv
    for task in cv_data:
        X_train, y_train, X_test, y_test = task['X_train'], task['y_train'], task['X_test'], task['y_test']
        X_train, y_train, X_test, y_test = X_train[features_names], y_train[target_name], \
                                           X_test[features_names], y_test[target_name]

        model.fit(X_train, y_train)

        predictions_test = model.predict(X_test)
        predictions_train = model.predict(X_train)

        test_data_to_evaluate = (predictions_test, y_test)
        train_data_to_evaluate = (predictions_train, y_train)

        scores_train = evaluate(*train_data_to_evaluate, metrics=metrics, data_set_name='train_')
        scores_test = evaluate(*test_data_to_evaluate, metrics=metrics)

        task_scores = {**scores_test, **scores_train}

        for score in task_scores.keys():
            if scores['raw_cv_scores'].get(score) is None:
                scores['raw_cv_scores'][score] = []

            scores['raw_cv_scores'][score].append(task_scores[score])

    # process cv scores
    summarized_cv_scores = summarize_scores(scores['raw_cv_scores'])
    scores['cv_scores'].update(summarized_cv_scores)
    scores.update(summarized_cv_scores)

    # final model
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], \
                                       data['X_test'], data['y_test']

    final_model = model.fit(X_train[features_names], y_train[target_name])

    predictions_test = model.predict(X_test[features_names])
    predictions_train = model.predict(X_train[features_names])

    test_data_to_evaluate = (predictions_test, y_test[target_name])
    train_data_to_evaluate = (predictions_train, y_train[target_name])

    scores_train = evaluate(*train_data_to_evaluate, metrics=metrics, data_set_name='train_')
    scores_test = evaluate(*test_data_to_evaluate, metrics=metrics)

    scores['test_scores'] = {**scores_test, **scores_train}

    experiment['scores'] = scores

    save_model(final_model, artifacts_path)
    save_model_scores(experiment['scores'], artifacts_path)


def get_parser():
    parser = argparse.ArgumentParser(description='Experiment params')
    parser.add_argument('--ix',
                        action='store',
                        type=int)

    parser.add_argument('--hyperparams',
                        action='store',
                        type=str)

    parser.add_argument('--artifacts_path',
                        action='store',
                        type=str)

    parser.add_argument('--metrics',
                        action='store',
                        type=str)

    return parser


def process_arg_parser(parser):
    args = parser.parse_args()

    experiment = {'ix': args.ix, 'hyperparams': json.loads(args.hyperparams)}

    artifacts_path = args.artifacts_path

    metrics = args.metrics

    kwargs = {'experiment': experiment, 'artifacts_path': artifacts_path, 'metrics': metrics}

    return kwargs


if __name__ == '__main__':
    my_parser = get_parser()

    kwargs = process_arg_parser(my_parser)

    main(**kwargs)

