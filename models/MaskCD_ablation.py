import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig
import torch

class cd_net(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model_name ="facebook/mask2former-swin-base-ade-semantic"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        model = Mask2FormerForUniversalSegmentation(config)
        # model = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_model_name,ignore_mismatched_sizes=True)
        decoder=model.model.pixel_level_module
        self.pixel_decoder=decoder
        self.classifier=nn.Conv2d(256,2,1,1)
        self.upsample=nn.Upsample(scale_factor=4,mode='bilinear')
    def forward(self,x1,x2):
        input=torch.cat([x1,x2])
        x=self.pixel_decoder(input)
        x=x.decoder_last_hidden_state
        x=self.classifier(x)
        x=self.upsample(x)
        return {"main_predictions":x}