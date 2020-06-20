import json
import os
import sys

# system paths
cwd = os.getcwd()
sys.path.append(cwd)

# internal
from labs.lab import LabManager
from tests.local_test.lgbm_reg.train import lgbm_reg
from tests.local_test.utils.secrets import slack_token


# globals  
config_path = 'tests/local_test/utils/experimenters_config/experimenters_config_test.json'



def run_lab(experiment_name):
    # experimenters callables
    experiments_callables = {experiment_name: lgbm_reg}

    # configs
    with open(config_path, 'r') as fp:
        slack_config = json.load(fp).get('slack_config')

    if slack_config:
        slack_config['slack_token'] = slack_token

    experiments_manager = LabManager(config_path=config_path, slack_config=slack_config)

    experiments_manager.run_experimenters(experiments_callables=experiments_callables, 
                                          subset=list(experiments_callables.keys()))


def test_skopt_searcher_local():
    experiment_name = 'lgbm_reg_dummy_bs'

    run_lab(experiment_name)


def test_grid_search_local():
    experiment_name = 'lgbm_reg_dummy_gs'

    run_lab(experiment_name)


def test_random_search_local():
    experiment_name = 'lgbm_reg_dummy_rs'

    run_lab(experiment_name)


if __name__ == '__main__':
    test_grid_search_local()
    test_random_search_local()
    test_skopt_searcher_local()





