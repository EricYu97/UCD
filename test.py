import torch
import model
import yaml
from utils import argparser
from transformers import AutoConfig

def main(args):

    dataset=None
    if args.dataset_name or args.dataset_path is not None:
        if args.dataset_name is None:
            raise ValueError("Please specify the dataset_name!")
        if args.dataset_name is None:
            raise ValueError("Please specify the dataset_path!")
        
        dataset={"dataset_name" : args.dataset_name ,"data_type" : "cloud" if args.cloud else "local", "dataset_path" : args.dataset_path}

    if args.external_config is not None:
        print("loading from external config")

        with open(args.external_config,'r') as f:
            configs=yaml.safe_load(f)
        print(configs)
        CD_framework=model.Change_Detection_Framework(config=configs)
    else:
        
        if dataset is not None:
            print(f"using specfied dataset {dataset}")

            CD_framework=model.Change_Detection_Framework.from_pretrained(args.model)
            for k,v in dataset.items():
                CD_framework.configs[k]=v
            # config=AutoConfig.from_pretrained(args.model)
            # config.update(dataset)
            # CD_framework=model.Change_Detection_Framework.from_pretrained(model_id, config=config)

        else:
            CD_framework=model.Change_Detection_Framework.from_pretrained(args.model)
    print("model loaded")
    CD_framework.configs["test"]["batch_size"]=args.batch_size # Reset batch_size here for testing here, default is 32
    CD_framework.testing_CD()

    if args.push_to_hub is not None:
        if args.external_config is not None:

            CD_framework.configs["test"]["use_external_checkpoint"]=False

        if CD_framework.accelerator.is_main_process:
            CD_framework.push_to_hub(repo_id=args.push_to_hub)


if __name__=="__main__":
    args=argparser.get_test_argparser().parse_args()

    # args=argparser.get_argparser().parse_args()

    if args.dataset_name or args.dataset_path is not None:
        if args.dataset_name is None:
            raise ValueError("Please specify the dataset_name!")
        if args.dataset_name is None:
            raise ValueError("Please specify the dataset_path!")

    main(args)

