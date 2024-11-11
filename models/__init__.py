import importlib
import torchvision
import copy

__all__=["A2Net","AERNet","BIT","ChangeFormer","CLAFA","DASNet","DMINet","DSAMNet","ICIFNet","RDPNet","Tiny_CD","USSFCNet","DSIFN","FC-EF","FCNPP","HANet","ResUnet","SiamUnet_diff","SNUNet","siamunet_conc"]

__unsupported_multi_class__=["AERNet","Tiny_CD","DSIFN"]
__unsupported_multi_input_nc__=["AERNet","CLAFA","DMINet","ICIFNet", "Tiny_CD", "USSFCNet","DSIFN"]
__not_implemented__=["DASNet"]
__sigmoid_model__=["A2Net","AERNet","Tiny_CD","USSFCNet","DSIFN","FCNPP","CGNet","HCGMNet"]

def find_model_using_name(configs):
    # assert model_name in __all__
    _warning(configs)
    model_name=configs["model_name"]
    print(f"loading {model_name}")
    if model_name== "A2Net":
        module=importlib.import_module('.A2Net.model',package='.models')
        model=module.BaseNet(input_nc=configs["input_nc"], output_nc=1)
    elif model_name== "AERNet":
        module=importlib.import_module('.AERNet.network',package='.models')
        model=module.zh_net()
    elif model_name== "BIT":
        module=importlib.import_module('.BIT.BIT_test',package='.models')
        model=module.BASE_Transformer(input_nc=configs["input_nc"], output_nc=configs["num_classes"], token_len=4, resnet_stages_num=4,
                            with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    elif model_name== "ChangeFormer":
        module=importlib.import_module('.ChangeFormer.ChangeFormer',package='.models')
        model=module.ChangeFormerV6(input_nc=configs["input_nc"], output_nc=configs["num_classes"])
    elif model_name== "CLAFA":
        module=importlib.import_module('.CLAFA.network',package='.models')
        model=module.Detector(num_classes=configs["num_classes"])
    elif model_name== "DASNet":
        NotImplementedError
        module=importlib.import_module(".DASNet.siameseNet.dares",package='.models')
        model=module.SiameseNet()
    elif model_name== "DMINet":
        module=importlib.import_module(".DMINet.DMINet",package='.models')
        model=module.DMINet(num_classes=configs["num_classes"])
    elif model_name== "DSAMNet":
        module=importlib.import_module(".DSAMNet.dsamnet",package='.models')
        model=module.DSAMNet(in_c=configs["input_nc"], n_class=configs["num_classes"])
    elif model_name== "ICIFNet":
        module=importlib.import_module(".ICIFNet.ICIFNet",package='.models')
        model=module.ICIFNet(num_classes=configs["num_classes"])
    elif model_name== "RDPNet":
        module=importlib.import_module(".RDPNet.RDPNet",package='.models')
        model=module.RDPNet(in_ch=configs["input_nc"],out_ch=configs["num_classes"])
    elif model_name== "Tiny_CD":
        module=importlib.import_module(".Tiny_CD.tinycd",package='.models')
        model=module.ChangeClassifier()
    elif model_name== "USSFCNet":
        module=importlib.import_module(".USSFCNet.USSFCNet",package='.models')
        model=module.USSFCNet(in_ch=configs["input_nc"], out_ch=1, ratio=0.5)
    elif model_name== "DSIFN":
        module=importlib.import_module(".DSIFN",package='.models')
        model=module.DSIFN()
    elif model_name== "FC-EF":
        module=importlib.import_module(".FC-EF",package='.models')
        model=module.Unet(input_nbr=configs["input_nc"], label_nbr=configs["num_classes"])
    elif model_name== "FCNPP":
        module=importlib.import_module(".FCNPP",package='.models')
        model=module.FCNPP(n_channels=configs["input_nc"], n_classes=1)
    elif model_name== "HANet":
        module=importlib.import_module(".HANet",package='.models')
        model=module.HAN(in_ch=configs["input_nc"], ou_ch=configs["num_classes"])
    elif model_name== "ResUnet":
        module=importlib.import_module('.ResUnet',package='.models')
        model=module.ResUnet(channel=configs["input_nc"], num_classes=configs["num_classes"])
    elif model_name== "siamunet_conc":
        module=importlib.import_module('.siamunet_conc',package='.models')
        model=module.SiamUnet_conc(input_nbr=configs["input_nc"], label_nbr=configs["num_classes"])
    elif model_name== "SiamUnet_diff":
        module=importlib.import_module('.SiamUnet_diff',package='.models')
        model=module.SiamUnet_diff(input_nbr=configs["input_nc"], label_nbr=configs["num_classes"])
    elif model_name== "SNUNet":
        module=importlib.import_module('.SNUNet',package='.models')
        model=module.SNUNet_ECAM(in_ch=configs["input_nc"], out_ch=configs["num_classes"])
    elif model_name== "MaskCD_ablation":
        module=importlib.import_module('.MaskCD_ablation',package='.models')
        model=module.cd_net()
    elif model_name== "AFCF3D":
        module=importlib.import_module('.AFCF3D',package='.models')
        resnet = torchvision.models.resnet18(pretrained=True)
        model = module.Netmodel(32, copy.deepcopy(resnet))
    elif model_name== "CGNet":
        module=importlib.import_module('.cgnet',package='.models')
        model = module.CGNet()
    elif model_name== "HCGMNet":
        module=importlib.import_module('.cgnet',package='.models')
        model = module.HCGMNet()
    elif model_name== "TFI_GR":
        module=importlib.import_module('.tfi_gr',package='.models')
        model = module.TFI_GR(input_nc=configs["input_nc"],output_nc=configs["num_classes"])
    elif model_name== "SwinSUnet":
        module=importlib.import_module('.swinsunet',package='.models')
        model = module.SwinSUNet(in_chans=configs["input_nc"], num_classes=configs["num_classes"]).cuda()
    elif model_name== "MSPSNet":
        module=importlib.import_module('.MSPSNet',package='.models')
        model = module.FEBlock1(in_ch=configs["input_nc"], ou_ch=configs["num_classes"]).cuda()
    elif model_name== "DTCDSCN":
        module=importlib.import_module('.DTCDSCN',package='.models')
        model = module.CDNet34(in_channels=configs["input_nc"],num_classes=configs["num_classes"]).cuda()
    elif model_name== "MineNetCD":
        module=importlib.import_module('.MineNetCD.minenetcd',package='.models')
        model=module.get_model_minenetcd(backbone_type=configs["model"]["backbone_type"] if "model" in configs and "backbone_type" in configs["model"] else "Swin_Diff_T",channel_mixing=True,num_classes=configs["num_classes"]).cuda()
    else:
        print("Please select one available model from the list:")
        print(__all__)

    # if model_name=="DMINet":
    #     module=importlib.import_module(".DMINet.DMINet",package='.models')
    #     model=module.DMINet()
    # elif model_name=="ResUnet":
    #     module=importlib.import_module('.ResUnet',package='.models')
    #     model=module.ResUnet()
    # elif model_name=="A2Net":
    #     module=importlib.import_module('.A2Net.model',package='.models')
    #     model=module.BaseNet()
    # elif model_name=="AERNet":
    #     pass
    # elif model_name=="BIT":
    #     pass
    # elif model_name=="ChangeFormer":
    #     pass
    # elif model_name=="CLAFA":
    #     pass

    return model

def _warning(configs):
    model_name=configs["model_name"]
    if model_name in __unsupported_multi_class__ and configs["num_classes"]>2:
        print(f"The implementation of {model_name} does not support multi-class change detection, please ensure the 'num_classes' is no more than 2.")
    if model_name in __unsupported_multi_input_nc__ and configs["input_nc"]>3:
        print(f"The implementation of {model_name} does not support multi-/ hyper-spectral change detection, please ensure the 'input_nc' is no more than 3.")
    if model_name in __not_implemented__:
        print(f"The model {model_name} has not been implemented yet, try another one.")
    if model_name in __sigmoid_model__:
        print(f"The model {model_name} contains a sigmoid activation at its end, the 'num_classes' has been set to the default value 1, please make sure using it for binary change detection.")
