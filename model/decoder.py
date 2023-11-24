import torch
import torch.nn as nn


# @brief: MLP with SDF classification
class MLP_reg(nn.Module):
    def __init__(self,
                 cfg,
                 input_ch=3,
                 input_ch_pos=12,
                 n_hidden=128,
                 n_hidden_rgb=64,
                 n_hidden_sdf=64,
                 n_hidden_branch=128,
                 n_class=5,
                 beta=80.):
        super(MLP_reg, self).__init__()
        self.cfg = cfg
        self.input_ch = input_ch  # dim of input parametric encoding
        self.input_ch_pos = input_ch_pos + 3  # dim of input positional encoding(including original xyz)
        self.n_hidden = n_hidden  # dim of hidden layers, default: 64
        self.n_hidden_rgb = n_hidden_rgb  # dim of RGB embedding, default: 32
        self.n_hidden_sdf = n_hidden_sdf  # dim of SDF embedding, default: 32
        self.n_hidden_branch = n_hidden_branch

        self.n_class = n_class  # number of classification, default: 5
        self.max_class_Id = self.n_class - 1
        self.beta = beta
        self.class_Ids = torch.arange(0., self.n_class, 1.).cuda()  # class_Id of each class (starting with 0), Tensor(class_num, )

        # first stage
        self.pts_linear = nn.Sequential(
            nn.Linear(self.input_ch_pos, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden_sdf + self.n_hidden_rgb)
        )

        # second stage - RGB branch
        self.rgb_linear = nn.Sequential(
            nn.Linear(self.n_hidden_rgb + self.input_ch_pos, 3),
        )

        # second stage - SDF branch
        self.sdf_linear = nn.Sequential(
            nn.Linear(self.n_hidden_sdf + self.input_ch, self.n_hidden_branch),
            nn.ReLU(),
            nn.Linear(self.n_hidden_branch, self.n_class),

            nn.Softmax(dim=-1)
        )


    def forward(self, embed, embed_pos, query_pts):
        embed_pos_w = torch.cat([query_pts, embed_pos], -1)  # PE including original xyz coordinates

        sdf_rgb = self.pts_linear(embed_pos_w)  # Tensor(N, 64)
        sdf_embedding = sdf_rgb[:, :self.n_hidden_sdf]  # Tensor(N, 32)
        rgb_embedding = sdf_rgb[:, self.n_hidden_sdf:]  # Tensor(N, 32)

        # branch 1: RGB
        h1 = torch.cat([rgb_embedding, embed_pos_w], dim=-1)
        rgb = self.rgb_linear(h1)  # Tensor(N, 3)

        # branch 2: SDF
        h2 = torch.cat([sdf_embedding, embed], dim=-1)
        sdf_prob = self.sdf_linear(h2)  # normalized classification probability, Tensor(N, class_num)

        sdf_entropy = -1. * torch.sum(sdf_prob * torch.log2(sdf_prob + 1e-5), dim=-1, keepdim=True)  # Tensor(N, 1)

        # pred probability --> SDF value
        sdf = torch.sum(sdf_prob * self.class_Ids[None, ...], dim=-1, keepdim=True)  # Tensor(N, 1)
        sdf = ( sdf / self.max_class_Id - 0.5 ) * 2  # to real truncated SDF value, [-1, 1], Tensor(N, 1)

        outputs = torch.cat([rgb, sdf, sdf_entropy, sdf_prob], dim=-1)  # Tensor(N, 10) (10=3+1+1+5)
        return outputs
