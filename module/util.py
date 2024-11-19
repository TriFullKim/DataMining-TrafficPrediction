import pickle
import datetime
import yaml


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_yaml(path, obj: dict):
    with open(path, "w") as f:
        yaml.dump(obj, f)


def now():
    return datetime.datetime.now().strftime("%F-%R")


def save_model_info(dir_path, model, time=now(),is_ensemble=True):
    __yaml = {}
    model_params_dict = model.get_params()
    if is_ensemble:
        model_params_dict["estimator"] = model_params_dict["estimator"].get_parmas()
    __yaml["param"] = model_params_dict
    __yaml["model_weight_path"] = f"{dir_path}-obj.pkl"
    __yaml["model_info_path"] = f"{dir_path}-info.yaml"
    __yaml["save_at"] = time
    __yaml["submission_file"]

    save_yaml(__yaml["model_info_path"], __yaml)
    save_pkl(__yaml["model_weight_path"], model)
