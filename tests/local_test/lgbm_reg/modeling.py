import joblib
import os


def create_model(hyperparams):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**hyperparams)


def save_model(model, model_save_path):
    joblib.dump(model, os.path.join(model_save_path, 'model.joblib'))