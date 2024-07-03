
from .trainer import MarigoldTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]