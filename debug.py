import torch
import model
import yaml
from utils import argparser

def main(configs):
    CD_framework=model.Change_Detection_Framework.from_pretrained("ericyu/GVLM256_DMINet")
    # CD_framework=model.Change_Detection_Framework(configs)
    CD_framework.debug()

if __name__=="__main__":
    args=argparser.get_argparser().parse_args()

    with open(args.config,'r') as f:
        configs=yaml.safe_load(f)
    main(configs)
    # print(configs)
    # main()

