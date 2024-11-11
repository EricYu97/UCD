import torch
import model
import yaml
from utils import argparser
from accelerate import utils

def main(configs):
    CD_framework=model.Change_Detection_Framework(config=configs)
    CD_framework.calculate_parameters()

if __name__=="__main__":
    utils.set_seed(8888)

    args=argparser.get_argparser().parse_args()

    with open(args.config,'r') as f:
        configs=yaml.safe_load(f)
        print(configs)
    main(configs)
    # print(configs)
    # main()

