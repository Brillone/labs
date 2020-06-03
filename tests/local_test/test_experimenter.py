import json

# internal
from labs.experimenting import LocalExperimenter
from tests.local_test.lgbm_reg.train import lgbm_reg
from tests.local_test.utils.secrets import slack_token


def run_experimenter(config_path):
    experiment_config = None
    slack_config = None

    # configs
    with open(config_path, 'r') as fp:
        experiments_config = json.load(fp)

    # lgbm_reg
    for experiment in experiments_config["experiments"]:
        if experiment.get('experiment_name') == 'lgbm_reg_dummy':
            experiment_config = experiment

            slack_config = experiments_config.get('slack_config')

            if slack_config is not None:
                slack_config['slack_token'] = slack_token

            break

    experiment = LocalExperimenter(**experiment_config,
                                   slack_config=slack_config,
                                   mode=experiments_config['mode'])

    experiment.run_experiments(lgbm_reg)


def test_skopt_searcher_local():
    # path skopt
    experiments_config_path = 'local_test/utils/configs/skopt_config_test.json'

    run_experimenter(experiments_config_path)


def test_grid_search_local():
    # path grid
    experiments_config_path = 'local_test/utils/configs/grid_config_test.json'

    run_experimenter(experiments_config_path)


def test_random_search_local():
    # path random
    experiments_config_path = 'local_test/utils/configs/random_config_test.json'

    run_experimenter(experiments_config_path)


if __name__ == '__main__':
    test_skopt_searcher_local()
    test_grid_search_local()
    test_random_search_local()







