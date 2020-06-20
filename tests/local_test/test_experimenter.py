import os
import sys
import json

# system paths
cwd = os.getcwd()
sys.path.append(cwd)

# internal
from labs.experimenting import LocalExperimenter
from tests.local_test.lgbm_reg.train import lgbm_reg
from tests.local_test.utils.secrets import slack_token


### globals ###
# evaluation
evaluation_config=dict(
    metrics=dict(MSE='min', MAE='min'),
    main_metric='MSE'
)

# dask
dask_config=dict(
    host='0.0.0.0',
    local_directory='/tmp/dask-worker-space',
    n_workers=4,
    threads_per_worker=2,
    processes=True
)

# slack config
slack_config = None
if slack_token:
    slack_config = {"recipient": '#channel_name', 'slack_token': slack_token}


def run_experimenter(experimnter_config, lab_config):
    experiment = LocalExperimenter(**experimnter_config, **lab_config)

    experiment.run_experiments(lgbm_reg)


def test_skopt_searcher_local():
    # space
    space_config=dict(
        n_estimators=dict(search_vals=[100, 2000], type='integer'),
        learning_rate=dict(search_vals=[0.0001, 0.1], type='real'),
        max_depth=dict(search_vals=[3, 5], type='categorical'),
        n_jobs=dict(search_vals=-1, type='static'),
        random_state=dict(search_vals=1234, type='static')
    )

    # tune
    tune_config = dict(
        search_params=dict(random_state=1234, base_estimator='GP', n_initial_points=5),
        space=space_config,
        n_experiments=40,
        experiments_batch_size=4,
        type='skopt',
        score_threshold=16900,
        delete_experiments=False
    )

    bs_experimenter_config = dict(
        run_id=None,
        experiment_name='lgbm_reg_dummy_bs',
        description='Dummy Regression using lgbm with bayesian guassian process search tuning',
        problem_name= 'Dummy Regression',
        artifacts_path= 'tests/local_test/artifacts',
        tune_config=tune_config,
        evaluation_config=evaluation_config,
        dask_config=dask_config
    )

    # path grid
    lab_config = dict(
        mode='research',
        slack_config=slack_config
    )

    run_experimenter(experimnter_config=bs_experimenter_config, lab_config=lab_config)


def test_grid_search_local():
    # space
    space_config=dict(
        n_estimators=dict(search_vals=[100, 250, 500, 1000, 2000], type='list'),
        learning_rate=dict(search_vals=[-4, -1], type='log-space', count=4),
        max_depth=dict(search_vals=[3, 5], type='list'),
        n_jobs=dict(search_vals=-1, type='static'),
        random_state=dict(search_vals=1234, type='static')
    )

    # tune
    tune_config = dict(
        space=space_config,
        experiments_batch_size=4,
        type='grid-search',
        score_threshold=16900
    )

    gs_experimenter_config = dict(
        run_id=None,
        experiment_name='lgbm_reg_dummy_gs',
        description='Dummy Regression using lgbm with grid search tuning',
        problem_name= 'Dummy Regression',
        artifacts_path= 'tests/local_test/artifacts',
        tune_config=tune_config,
        evaluation_config=evaluation_config,
        dask_config=dask_config
    )

    # path grid
    lab_config = dict(
        mode='research',
        slack_config=slack_config
    )

    run_experimenter(experimnter_config=gs_experimenter_config, lab_config=lab_config)


def test_random_search_local():
    # space
    space_config=dict(
        n_estimators=dict(search_vals=[100, 2000], type='int-uniform'),
        learning_rate=dict(search_vals=[0.1], type='exp'),
        max_depth=dict(search_vals=[3, 5], type='categorical'),
        n_jobs=dict(search_vals=-1, type='static'),
        random_state=dict(search_vals=1234, type='static')
    )

    # tune
    tune_config = dict(
        search_params=dict(random_state=1234),
        space=space_config,
        n_experiments=40,
        experiments_batch_size=4,
        type='random-search',
        score_threshold=16900,
        delete_experiments=False
    )

    rs_experimenter_config = dict(
        run_id=None,
        experiment_name='lgbm_reg_dummy_rs',
        description='Dummy Regression using lgbm with random search tuning',
        problem_name= 'Dummy Regression',
        artifacts_path= 'tests/local_test/artifacts',
        tune_config=tune_config,
        evaluation_config=evaluation_config,
        dask_config=dask_config
    )

    # path grid
    lab_config = dict(
        mode='research',
        slack_config=slack_config
    )

    run_experimenter(experimnter_config=rs_experimenter_config, lab_config=lab_config)


if __name__ == '__main__':
    test_grid_search_local()
    test_random_search_local()
    test_skopt_searcher_local()







