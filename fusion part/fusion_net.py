import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from fusion_strategy import MAFusion

class Learnable_Fusion_mask_before(nn.Module):
    def __init__(self, channel_num):
        super(Learnable_Fusion_mask_before, self).__init__()
        self.le_fusion_fore1 = MAFusion(channel_num[0])
        self.le_fusion_fore2 = MAFusion(channel_num[1])
        self.le_fusion_fore3 = MAFusion(channel_num[2])
        self.le_fusion_fore4 = MAFusion(channel_num[3])

        # 或许fore用MA，back用CD也可以
        self.le_fusion_back1 = MAFusion(channel_num[0])
        self.le_fusion_back2 = MAFusion(channel_num[1])
        self.le_fusion_back3 = MAFusion(channel_num[2])
        self.le_fusion_back4 = MAFusion(channel_num[3])

    def forward(self, en_vis, en_ir, mask_list):
        f1_fore = self.le_fusion_fore1(en_ir[0] * mask_list[0], en_vis[0] * mask_list[0])
        f1_back = self.le_fusion_back1(en_vis[0] * (1 - mask_list[0]), en_ir[0] * (1 - mask_list[0]))
        f2_fore = self.le_fusion_fore2(en_ir[1] * mask_list[1], en_vis[1] * mask_list[1])
        f2_back = self.le_fusion_back2(en_vis[1] * (1 - mask_list[1]), en_ir[1] * (1 - mask_list[1]))
        f3_fore = self.le_fusion_fore3(en_ir[2] * mask_list[2], en_vis[2] * mask_list[2])
        f3_back = self.le_fusion_back3(en_vis[2] * (1 - mask_list[2]), en_ir[2] * (1 - mask_list[2]))
        f4_fore = self.le_fusion_fore4(en_ir[3] * mask_list[3], en_vis[3] * mask_list[3])
        f4_back = self.le_fusion_back4(en_vis[3] * (1 - mask_list[3]), en_ir[3] * (1 - mask_list[3]))
        f1 = f1_fore + f1_back
        f2 = f2_fore + f2_back
        f3 = f3_fore + f3_back
        f4 = f4_fore + f4_back
        return [f1, f2, f3, f4]