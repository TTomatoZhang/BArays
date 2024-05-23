import logging
import os
import os.path as osp
import sys
import math

import yaml

# try:
#     sys.path.append("../../")
#     from utils import OrderedYaml
# except ImportError:
#     pass
sys.path.append("../../")
from utils import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)
    name = opt['name']


    dataset = opt["dataset"]
    if dataset.get("root", None) is not None:
        dataset["root"] = osp.expanduser(dataset["root"])

    # path
    for key, path in opt["path"].items():
        if path and key in opt["path"] and key != "strict_load":
            opt["path"][key] = osp.expanduser(path)
    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir,) #  osp.pardir, osp.pardir, osp.pardir)
    )
    path = osp.abspath(__file__)

    if is_train:
        dataname = name
        experiments_root = osp.join(
            "experiments", dataname, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")



        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        dataname = 'E-sde4DEM-x2-gradloss-forx2'
        results_root = osp.join(opt["path"]["root"], "results", name)
        opt["path"]["results_root"] = osp.join(results_root, opt["name"])
        opt["path"]["log"] = osp.join(results_root, opt["name"])

    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths"""
    logger = logging.getLogger("base")
    if opt["path"]["resume_state"]:
        if (
            opt["path"].get("pretrain_model_G", None) is not None
            or opt["path"].get("pretrain_model_D", None) is not None
        ):
            logger.warning(
                "pretrain_model path will be ignored when resuming training."
            )

        opt["path"]["pretrain_model_G"] = osp.join(
            opt["path"]["models"], "{}_G.pth".format(resume_iter)
        )
        logger.info("Set [pretrain_model_G] to " + opt["path"]["pretrain_model_G"])
        if "gan" in opt["model"]:
            opt["path"]["pretrain_model_D"] = osp.join(
                opt["path"]["models"], "{}_D.pth".format(resume_iter)
            )
            logger.info("Set [pretrain_model_D] to " + opt["path"]["pretrain_model_D"])
