<div align="center">
<h1>Unified Change Detection Framework </h1>
<h3>Powered by Huggingface Hub 🤗 </h3> 
</div>

### Contributors:
[Weikang Yu](https://ericyu97.github.io/), [Xiaokang Zhang](https://xkzhang.info/), [Richard Gloaguen](https://scholar.google.de/citations?user=e1QDLQUAAAAJ&hl=de), [Xiao Xiang Zhu](https://www.asg.ed.tum.de/sipeo/home/), [Pedram Ghamisi](https://www.ai4rs.com/)

## News
``11.11.2024`` UCD is open to everyone! Be a contributor by sending a pull request!

``11.11.2024`` Codes for UCD have been released, if you find any problems or bugs, please leave us a message.

``09.11.2024`` Our paper of MineNetCD has been published on IEEE TGRS 2024, the repo for MineNetCD is available [here](https://github.com/AI4RS/MineNetCD).

``09.07.2024`` The UCD project is announced on IEEE IGARSS 2024, we are organizing the codes.
## Environment Preparation:
Create a conda environment for UCD
 ```console
conda create -n ucd python=3.10
conda activate ucd
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Configurate the accelerate package:
```console
accelerate config
```

## How to use:
### To train a model:
```bash
accelerate launch train.py --config $CONFIG$
```
An example of CONFIG can be ```configs/DTCDSCN_MNCD256.yml```

Or if you want to run the framework in Singularity & Slurm, you can use this command:

```
srun singularity exec --env PREPEND_PATH=/home/yu34/.local/bin --nv /home/yu34/ucd.sif accelerate launch train.py --config $CONFIG$
```

---
### To test a model:
```bash
accelerate launch test.py --model $PRETRAINED_MODEL_PATH$ 
```
An example of PRETRAINED_MODEL_PATH can be a local path like ```checkpoints/MNCD256/ResUnet/BestF1/``` or a Huggingface hub id like ```HZDR-FWGEL/UCD-MNCD256-ResUnet```

### To push a pretrained model to Huggingface hub in the UCD format:
```bash
accelerate launch test.py --model $LOCAL_PATH$ --batch-size 10 --push-to-hub $HUB_NAME$
```
The example of LOCAL_PATH can be ```checkpoints/MNCD256/ResUnet/BestF1/```, the example of HUB_NAME can be ```HZDR-FWGEL/UCD-MNCD256-ResUnet```

---
To transfer a model from torch format (e.g., pytorch_model.bin) to UCD compatible format:

You should prepare a config file in your directory, an example can be found via configs/MineNetCD_MNCD256.yml

```bash
accelerate launch test.py --external-config configs/MineNetCD_MNCD256.yml  --batch-size 10 --push-to-hub HZDR-FWGEL/UCD-MNCD256-ChangeFFT
```
The ```use_external_checkpint``` will be set to false before uploading the model to the hub.

### To calculate model parameters & FLOPs:

```bash
python params_sum.py --config $CONFIG$
```

---
## Available Models:


<div align="center">

| Model   | Backbone   |\#Params| FLOPs    |Source| 
| :---:   | :---:   | :---:   | :---:   | :---:  |
|[A2Net](https://github.com/guanyuezhen/A2Net) | VGG16 | 3.60M | 2.86G | IEEE TGRS 2023 |
|[AFCF3D](https://github.com/wm-Githuber/AFCF3D-Net) | ResNet18 | 16.83M | 29.54G | IEEE TGRS 2023 |
|[BIT](https://github.com/justchenhao/BIT_CD) | ResNet18 | 11.39M | 8.28G | IEEE TGRS 2021 |
|[CGNet](https://github.com/ChengxiHAN/CGNet-CD) | VGG16 | 37.18M | 81.66G | IEEE JSTARS 2023 |
|[ChangeFormer](https://github.com/wgcban/ChangeFormer) | MIT | 39.13M | 129.7G | IEEE IGARSS 2022 |
|[DMINet](https://github.com/ZhengJianwei2/DMINet) | ResNet18 | 6.44M | 16.16G | IEEE TGRS 2021 |
|[DTCDSCN](https://github.com/fitzpchao/DTCDSCN)  | - | 39.17M | 10.72G | IEEE GRSL 2020 |
|[FC-EF](https://ieeexplore.ieee.org/document/9554522) | - | 1.29M | 2.92G | IEEE ICIP 2018 |
|[FCNPP](https://ieeexplore.ieee.org/document/8618401) | - | 14.56M | 43.10G | IEEE GRSL 2019 |
|[HCGMNet](https://github.com/ChengxiHAN/CGNet-CD) | VGG16 | 45.13M | 301.82G | IEEE IGARSS 2023 |
|[ICIFNet](https://github.com/ZhengJianwei2/ICIF-Net) | ResNet18 | 24.64M | 22.97G | IEEE TGRS 2022 |
|[MSPSNet](https://github.com/QingleGuo/MSPSNet-Change-Detection-TGRS) | - | 2.11M | 13.89G | IEEE TGRS 2021 |
|[RDPNet](https://github.com/Chnja/RDPNet) | - | 1.62M | 1.63G | IEEE TGRS 2022 |
|[ResUnet](https://ieeexplore.ieee.org/document/9553995) | - | 12.59M | 28.51G | IEEE IGARSS 2021 |
|[SiamUnet-Conc](https://github.com/rcdaudt/fully_convolutional_change_detection) | - | 1.47M | 4.55G | IEEE ICIP 2018 |
|[SiamUnet-Diff](https://github.com/rcdaudt/fully_convolutional_change_detection) | - | 1.29M | 3.99G | IEEE ICIP 2018 |
|[SNUNet](https://github.com/likyoo/Siam-NestedUNet) | - | 11.48M | 43.6G | IEEE GRSL 2021 |
|[TFI-GR](https://github.com/guanyuezhen/TFI-GR)  | ResNet18 | 27.06M | 9.09G | IEEE TGRS 2022 |
|[TinyCD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) | EfficientNet-b4 | 0.27M | 1.44G | NCA 2023 |
|[MineNetCD](https://github.com/AI4RS/MineNetCD)| SwinT-Tiny | 57.81M | 63.28G | This Paper | 

</div>

## Available Datasets:

<div align="center">

|Dataset   | \#Patches | Scenario  | Location         | Sensor       | Resolution | 
| :---:   | :---:   | :---:   | :---:   | :---:  | :---:  |
|[CLCD](https://github.com/liumency/CropLand-CD)  | 2400      | Cropland  | Guangdong, China | Gaofen-2     | 0.5m-2m     |
|[EGY-BCD](https://github.com/oshholail/EGY-BCD)  | 6091      | Building  | Egypt            | Google Earth | 0.25m      |
|[GVLM-CD](https://github.com/zxk688/GVLM)   | 7496      | Landslide | Global           | Google Earth | 0.59m      |
|[LEVIR-CD](https://chenhao.in/LEVIR/)  | 10192     | Building  | Texas, USA       | Google Earth | 0.5m       |
|[SYSU-CD](https://github.com/liumency/SYSU-CD)   | 20000     | Urban     | Hong Kong, China | Aerial Image      | 0.5m       | 
|[MineNetCD](https://github.com/AI4RS/MineNetCD)| 71711     | Mining    | Global           | Google Earth | 1.2m       | 
</div>

## Our Results:
Here are results derived from the UCD 

### CLCD256:

Dataset for these implementations: ``ericyu/CLCD_Cropped_256``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | CLCD256| 0.9199     | 0.3765     |0.4474    | 0.3250  | 0.2319 | ``HZDR-FWGEL/UCD-CLCD256-A2Net``|
| BIT | CLCD256     | 0.9488    |0.6590    | 0.6526  | 0.6657 | 0.4915 | ``HZDR-FWGEL/UCD-CLCD256-BIT``|
| DMINet | CLCD256     | 0.9392   |0.5744   | 0.5990| 0.5517 | 0.4029 | ``HZDR-FWGEL/UCD-CLCD256-DMINet``|
| ICIFNet | CLCD256     | 0.9416   |0.5629    | 0.6355  | 0.5052 | 0.3917 | ``HZDR-FWGEL/UCD-CLCD256-ICIFNet``|
| RDPNet | CLCD256    | 0.9288   |0.5431   | 0.5194 | 0.569 | 0.3727 | ``HZDR-FWGEL/UCD-CLCD256-RDPNet``|
| SiamUNet-Diff | CLCD256    | 0.9358  |0.4914 | 0.5983 |0.4169 | 0.3257 | ``HZDR-FWGEL/UCD-CLCD256-SiamUDiff``|
| ChangeFormer | CLCD256    | 0.9431  |0.6214 | 0.6151 |0.6279| 0.4508 | ``HZDR-FWGEL/UCD-CLCD256-ChangeFormer``|

</div>

### GVLM256:

Dataset for these implementations: ``ericyu/GVLM_Cropped256``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | GVLM256     | 0.9776    |0.8114    | 0.9156  | 0.7285 | 0.6827 |``HZDR-FWGEL/UCD-GVLM256-A2Net``|
| BIT | GVLM256     | 0.9841    |0.8768    |0.8974| 0.8572 | 0.7807 | ``HZDR-FWGEL/UCD-GVLM256-BIT``|
| DMINet | GVLM2256     | 0.9825   |0.8664    | 0.8738| 0.8591 | 0.7643| ``HZDR-FWGEL/UCD-GVLM256-DMINet``|
| ICIFNet | GVLM256     | 0.9831   |0.8722   | 0.8735  | 0.871| 0.7734 | ``HZDR-FWGEL/UCD-GVLM256-ICIFNet``|
| RDPNet | GVLM256    | 0.9827   |0.868   | 0.875 |0.8611 | 0.7668 | ``HZDR-FWGEL/UCD-GVLM256-RDPNet``|
| SiamUNet-Diff | GVLM256    | 0.9801  |0.8431  | 0.8791 |0.81 | 0.7288 | ``HZDR-FWGEL/UCD-GVLM256-SiamUDiff``|
| ChangeFormer | GVLM256    | 0.9831  |0.8685 | 0.8943 |0.8441| 0.7675 | ``HZDR-FWGEL/UCD-GVLM256-ChangeFormer``|

</div>

### EGYBCD:

Dataset for these implementations: ``ericyu/EGY_BCD``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | EGY_BCD     | 0.9624    |0.6914    | 0.7283  | 0.6581 | 0.5284 | ``HZDR-FWGEL/UCD-EGYBCD-A2Net``|
| BIT | EGYBCD     | 0.9735    |0.7906    | 0.8016  | 0.7799 | 0.6537 | ``HZDR-FWGEL/UCD-EGYBCD-BIT``|
| DMINet | EGYBCD     | 0.9585   |0.6929   | 0.6591| 0.7304 | 0.5301 | ``HZDR-FWGEL/UCD-EGYBCD-DMINet``|
| ICIFNet | EGYBCD     | 0.9621   |0.6903    | 0.7241  | 0.6595 | 0.5270 | ``HZDR-FWGEL/UCD-EGYBCD-ICIFNet``|
| RDPNet | EGYBCD    | 0.9612   |0.6859   | 0.7125 |0.6612 | 0.5220 | ``HZDR-FWGEL/UCD-EGYBCD-RDPNet``|
| SiamUNet-Diff | EGYBCD    | 0.9524 |0.6422 | 0.6191 | 0.6671 | 0.4729 | ``HZDR-FWGEL/UCD-EGYBCD-SiamUDiff``|
| ChangeFormer | EGYBCD    | 0.9651 |0.7181 | 0.7436 |0.6944| 0.5602 | ``HZDR-FWGEL/UCD-EGYBCD-ChangeFormer``|

</div>

### LEVIRCD256:

Dataset for these implementations: ``ericyu/LEVIRCD_Cropped256``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | LEVIRCD256     | 0.9699    |0.6687    | 0.7613  | 0.5962 | 0.5023 |``HZDR-FWGEL/UCD-LEVIRCD256-A2Net``|
| BIT | LEVIRCD256     | 0.9888    |0.8884    |0.9046 | 0.8728 | 0.7992 | ``HZDR-FWGEL/UCD-LEVIRCD256-BIT``|
| DMINet | LEVIRCD256     | 0.9845   |0.8431    | 0.8708| 0.8171 | 0.7287 | ``HZDR-FWGEL/UCD-LEVIRCD256-DMINet``|
| ICIFNet | LEVIRCD256     | 0.9827   |0.8162   | 0.8871  | 0.7558 | 0.6895 | ``HZDR-FWGEL/UCD-LEVIRCD256-ICIFNet``|
| RDPNet | LEVIRCD256    | 0.9808  |0.8058  | 0.8315 |0.7816 | 0.6747 | ``HZDR-FWGEL/UCD-LEVIRCD256-RDPNet``|
| SiamUNet-Diff | LEVIRCD256    | 0.9805 |0.5991  | 0.7822 |0.6874| 0.6423 | ``HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff``|
| ChangeFormer | LEVIRCD256    | 0.9826 |0.8232 | 0.8516 |0.7967| 0.6996 | ``HZDR-FWGEL/UCD-LEVIRCD256-ChangeFormer``|

</div>

### SYSUCD:

Dataset for these implementations: ``ericyu/SYSU_CD``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | SYSUCD     | 0.8812    |0.7598    | 0.7260  | 0.7969 | 0.6126 |``HZDR-FWGEL/UCD-SYSUCD-A2Net``|
| BIT | SYSUCD    | 0.873    |0.7497    |0.7004| 0.8064| 0.5996 | ``HZDR-FWGEL/UCD-SYSUCD-BIT``|
| DMINet | SYSUCD     | 0.8881   |0.7464    | 0.8014| 0.6984 | 0.5954 | ``HZDR-FWGEL/UCD-SYSUCD-DMINet``|
| ICIFNet | SYSUCD     | 0.8640|0.703   | 0.7248  | 0.6825 | 0.5421 | ``HZDR-FWGEL/UCD-SYSUCD-ICIFNet``|
| RDPNet | SYSUCD    | 0.8852   |0.7536  | 0.763 |0.7445| 0.6047 | ``HZDR-FWGEL/UCD-SYSUCD-RDPNet``|
| SiamUNet-Diff | SYSUCD    | 0.8546 |0.5991  | 0.8563 |0.4608| 0.4277 | ``HZDR-FWGEL/UCD-SYSUCD-SiamUDiff``|
| ChangeFormer | SYSUCD   | 0.8912 |0.7593 | 0.7938 |0.7277 | 0.612 | ``HZDR-FWGEL/UCD-SYSUCD-ChangeFormer``|

</div>

### MineNetCD256:

Dataset for these implementations: ``HZDR-FWGEL/MineNetCD256``

<div align="center">

| Model   | Dataset   |Accuracy| mF1     |Precision| Recall | cIoU  | Pretrained_Path|
| :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---: | :---|
| A2Net | MineNetCD256     | 0.9185    |0.6404    | 0.7215  | 0.5758 | 0.4710 | ``HZDR-FWGEL/UCD-MNCD256-A2Net``|
| AFCF3D | MineNetCD256     | 0.8932    |0.5772    | 0.5755  | 0.5789 | 0.4061 | ``HZDR-FWGEL/UCD-MNCD256-AFCF3D``*|
| BIT | MineNetCD256     | 0.9115    |0.6227    | 0.6727  | 0.5795 | 0.4521 | ``HZDR-FWGEL/UCD-MNCD256-BIT``|
| ChangeFormer | MineNetCD256     | 0.8699   |0.4995   | 0.4848  | 0.5151 | 0.3329 | ``HZDR-FWGEL/UCD-MNCD256-ChangeFormer``|
| CGNet | MineNetCD256     | 0.9004    |0.547    | 0.6412  | 0.477 | 0.3765 | ``HZDR-FWGEL/UCD-MNCD256-CGNet``|
| DMINet | MineNetCD256     | 0.8963   |0.5169    | 0.6257  | 0.4403 | 0.3485 | ``HZDR-FWGEL/UCD-MNCD256-DMINet``|
| DTCDSCN | MineNetCD256     | 0.8984   |0.5567    | 0.6184  | 0.5068 | 0.3864 | ``HZDR-FWGEL/UCD-MNCD256-DTCDSCN``*|
| FC-EF | MineNetCD256     | 0.8836   |0.415    | 0.5625  | 0.329 | 0.2619 | ``HZDR-FWGEL/UCD-MNCD256-FCEF``*|
| FCNPP | MineNetCD256     | 0.8549   |0.3449   | 0.4004  | 0.3030 | 0.2084 | ``HZDR-FWGEL/UCD-MNCD256-FCNPP``|
| HCGMNet | MineNetCD256     | 0.9076   |0.5876  | 0.6718  | 0.5222 | 0.4161 | ``HZDR-FWGEL/UCD-MNCD256-HCGMNet``|
| ICIFNet | MineNetCD256     | 0.8915   |0.5018    | 0.5958  | 0.4334 | 0.3349 | ``HZDR-FWGEL/UCD-MNCD256-ICIFNet``|
| ChangeFFT | MineNetCD256 | 0.9251 | 0.6963 | 0.7120 | 0.6814 | 0.5343 | ``HZDR-FWGEL/UCD-MNCD256-ChangeFFT``|
| MSPSNet | MineNetCD256     | 0.8998   |0.5591    | 0.6277  | 0.5041 | 0.388| ``HZDR-FWGEL/UCD-MNCD256-MSPSNet``|
|RDPNet | MineNetCD256     | 0.8768   |0.4961    | 0.5120  | 0.4811 | 0.3298 | ``HZDR-FWGEL/UCD-MNCD256-RDPNet``|
|ResUnet | MineNetCD256     | 0.8663   |0.5072    | 0.4727  | 0.5488 | 0.3398 | ``HZDR-FWGEL/UCD-MNCD256-ResUnet``|
| SiamUNet-Conc | MineNetCD256     | 0.8979   |0.5099    | 0.6460  | 0.4211 | 0.3422 | ``HZDR-FWGEL/UCD-MNCD256-SiamUConc``|
| SiamUNet-Diff | MineNetCD256     | 0.8956   |0.3736    | 0.7671  | 0.2469 | 0.2297 | ``HZDR-FWGEL/UCD-MNCD256-SiamUDiff``|
| SNUNet | MineNetCD256   | 0.8988  | 0.5371   |0.6351    | 0.4654  | 0.3675 | ``HZDR-FWGEL/UCD-MNCD256-SNUNet``*|
| TFI_GR | MineNetCD256     | 0.8932   |0.5772    |0.5755  | 0.5789 | 0.4061 | ``HZDR-FWGEL/UCD-MNCD256-TFIGR``*|
| TinyCD | MineNetCD256     | 0.8999   |0.5648    | 0.625  | 0.5153 | 0.3936 | ``HZDR-FWGEL/UCD-MNCD256-TinyCD``|

</div>
* Pretrained Models may only be loaded using accelerator with multiple graphical cards.

## Tutorial Avaiable!
We just added a very simple example as a tutorial for those who are interested in change detection, check [here](https://github.com/EricYu97/CDTutorial) for more details.


## Future Development Schedule:

We will implement more models and datasets. If you are interested in this project and want to make any contributions, please send a pull request and we will add your names under the contributors!

If you have any questions or meeting any difficulties when using this framework, please leave us with an issue or you can contact us with email address: [w.yu@hzdr.de](mailto:w.yu@hzdr.de)

## Acknowledgement:

We would like to thank Huggingface for providing a wonderful open-source platform. We would also like to thank all the authors and contributors who open-sourced the datasets and models that we incorporated into the UCD platform.

## Citation

If you find MineNetCD useful for your study, please kindly cite us:
```
@ARTICLE{10744421,
  author={Yu, Weikang and Zhang, Xiaokang and Gloaguen, Richard and Zhu, Xiao Xiang and Ghamisi, Pedram},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MineNetCD: A Benchmark for Global Mining Change Detection on Remote Sensing Imagery}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Data mining;Remote sensing;Feature extraction;Benchmark testing;Earth;Transformers;Annotations;Graphical models;Distribution functions;Sustainable development;Mining change detection;remote sensing;benchmark;frequency domain learning;unified framework},
  doi={10.1109/TGRS.2024.3491715}}
```