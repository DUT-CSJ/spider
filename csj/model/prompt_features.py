import timm
import torch
import torch.nn as nn

class convnext_fea(nn.Module):
    def __init__(self):
        super().__init__()
        # timm.list
        ###############################Transition Layer#######################################
        self.bkbone_prompt = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)


    def forward(self, x):
        with torch.no_grad():
            self.bkbone_prompt.eval()
            _, _, _, feat = self.bkbone_prompt(x)  # B,C,H,W
        return feat

class swin_fea(nn.Module):
    def __init__(self):
        super().__init__()
        # timm.list
        ###############################Transition Layer#######################################
        self.bkbone = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', features_only=True,
                                        pretrained=True)


    def forward(self, x):
        with torch.no_grad():
            self.bkbone_prompt.eval()
            _, _, _, feat = self.bkbone_prompt(x)  # B,C,H,W
            feat = feat.permute(0, 3, 1, 2)
        return feat