import argparse

def get_argparser():
    parser = argparse.ArgumentParser(
                    prog='Change_Detection_Framework',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--config',type=str,required=True,help="Directory of the config file")
    return parser

def get_test_argparser():
    parser = argparse.ArgumentParser(
                    prog='Change_Detection_Framework',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--model',type=str,default=None, help="Pretrained model path, or Huggingface Model ID")
    parser.add_argument('--dataset-path',type=str,default=None,help="Dataset path used for testing")
    parser.add_argument('--dataset-name',type=str,default=None,help="Dataset name used for testing")
    parser.add_argument('--cloud',type=bool, default=False, help="Whether model is stored on cloud")

    parser.add_argument('--push-to-hub',type=str, default=None,help="Push the model to hub with the repo id provided")

    parser.add_argument('--external-config',type=str,default=None,help="Using external config for testing, external pretrained models can be used.")

    parser.add_argument('--batch-size',type=int,default=32,help="Batch-size for testing")
    return parser

