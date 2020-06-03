import json

# internal
from labs.lab import LabManager
from tests.local_test.lgbm_reg.train import lgbm_reg
from tests.local_test.utils.secrets import slack_token


def run_lab(config_path):
    experiments_callables = {'lgbm_reg_dummy': lgbm_reg}

    # configs
    with open(config_path, 'r') as fp:
        slack_config = json.load(fp).get('slack_config')

    if slack_config is not None:
        slack_config['slack_token'] = slack_token

    experiments_manager = LabManager(config_path=config_path, slack_config=slack_config)

    experiments_manager.run_experimenters(experiments_callables)


def test_skopt_searcher_local():
    # path skopt
    experiments_config_path = 'local_test/utils/configs/skopt_config_test.json'

    run_lab(experiments_config_path)


def test_grid_search_local():
    # path grid
    experiments_config_path = 'local_test/utils/configs/grid_config_test.json'

    run_lab(experiments_config_path)


def test_random_search_local():
    # path random
    experiments_config_path = 'local_test/utils/configs/random_config_test.json'

    run_lab(experiments_config_path)


if __name__ == '__main__':
    test_skopt_searcher_local()
    test_grid_search_local()
    test_random_search_local()





