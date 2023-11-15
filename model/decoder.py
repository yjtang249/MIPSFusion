import torch
import torch.nn as nn
import tinycudann as tcnn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch  # default: 48
        self.geo_feat_dim = geo_feat_dim  # default: 15
        self.hidden_dim_color = hidden_dim_color  # default: 32
        self.num_layers_color = num_layers_color  # default: 2

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))


class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch  # default: 80 (32 + 48)
        self.geo_feat_dim = geo_feat_dim  # default: 15
        self.hidden_dim = hidden_dim  # default: 32
        self.num_layers = num_layers  # default: 2

        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])
    
    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:  # default
            sdf_net = []
            for l in range(self.num_layers):  # default: 2
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim 
                
                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim 
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))


class ColorSDFNet_v2(nn.Module):
    '''
    No color grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v2, self).__init__()
        self.config = config
        self.color_net = ColorNet(config, 
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])


    def forward(self, embed, embed_pos):
        if embed_pos is not None:  # default (for SDF)
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[..., :1], h[..., 1:]
        if embed_pos is not None:  # default (for color)
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)

############################################ My network ############################################
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
            # nn.Linear(self.n_hidden_rgb + self.input_ch, self.n_hidden_branch),
            # nn.ReLU(),
            # nn.Linear(self.n_hidden_branch, 3),

            nn.Linear(self.n_hidden_rgb + self.input_ch_pos, 3),
        )

        # second stage - SDF branch
        self.sdf_linear = nn.Sequential(
            nn.Linear(self.n_hidden_sdf + self.input_ch, self.n_hidden_branch),
            nn.ReLU(),
            nn.Linear(self.n_hidden_branch, self.n_class),

            # nn.Linear(self.n_hidden_sdf + self.input_ch, self.n_class),
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


# # @brief: light MLP
# class MLP_reg(nn.Module):
#     def __init__(self,
#                  cfg,
#                  input_ch=3,
#                  input_ch_pos=12,
#                  n_hidden=128,
#                  n_hidden_rgb=32,
#                  n_hidden_sdf=32,
#                  n_hidden_branch=32,
#                  n_class=5,
#                  beta=80.):
#         super(MLP_reg, self).__init__()
#         self.cfg = cfg
#         self.input_ch = input_ch  # dim of input parametric encoding
#         self.input_ch_pos = input_ch_pos  # dim of input positional encoding(including original xyz)
#         self.n_hidden = n_hidden  # dim of hidden layers, default: 64
#         self.n_hidden_rgb = n_hidden_rgb  # dim of RGB embedding, default: 32
#         self.n_hidden_sdf = n_hidden_sdf  # dim of SDF embedding, default: 32
#         self.n_hidden_branch = n_hidden_branch
#
#         self.n_class = n_class  # number of classification, default: 5
#         self.max_class_Id = self.n_class - 1
#         self.beta = beta
#         self.class_Ids = torch.arange(0., self.n_class, 1.).cuda()  # class_Id of each class (starting with 0), Tensor(class_num, )
#
#         # first stage
#         self.pts_linear = nn.Sequential(
#             nn.Linear(self.input_ch + self.input_ch_pos, self.n_hidden),
#             nn.ReLU(),
#             nn.Linear(self.n_hidden, self.n_hidden_sdf + self.n_hidden_rgb)
#         )
#
#         # second stage - RGB branch
#         self.rgb_linear = nn.Sequential(
#             # nn.Linear(self.n_hidden_rgb + self.input_ch, self.n_hidden_branch),
#             # nn.ReLU(),
#             # nn.Linear(self.n_hidden_branch, 3),
#
#             nn.Linear(self.n_hidden_rgb + self.input_ch_pos, 3),
#         )
#
#         # second stage - SDF branch
#         self.sdf_linear = nn.Sequential(
#             # nn.Linear(self.n_hidden_sdf + self.input_ch, self.n_hidden_branch),
#             # nn.ReLU(),
#             # nn.Linear(self.n_hidden_branch, self.n_class),
#
#             nn.Linear(self.n_hidden_sdf, self.n_class),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, embed, embed_pos, query_pts):
#         embed_pos_w = torch.cat([query_pts, embed_pos], -1)  # PE including original xyz coordinates
#         inputs = torch.cat([embed_pos, embed], dim=-1)
#
#         sdf_rgb = self.pts_linear(inputs)  # Tensor(N, 64)
#         sdf_embedding = sdf_rgb[:, :self.n_hidden_sdf]  # Tensor(N, 32)
#         rgb_embedding = sdf_rgb[:, self.n_hidden_sdf:]  # Tensor(N, 32)
#
#         # branch 1: RGB
#         h1 = torch.cat([rgb_embedding, embed_pos], dim=-1)
#         rgb = self.rgb_linear(h1)  # Tensor(N, 3)
#
#         # branch 2: SDF
#         sdf_prob = self.sdf_linear(sdf_embedding)  # normalized classification probability, Tensor(N, class_num)
#
#         sdf_entropy = -1. * torch.sum(sdf_prob * torch.log2(sdf_prob + 1e-5), dim=-1, keepdim=True)  # Tensor(N, 1)
#
#         # pred probability --> SDF value
#         sdf = torch.sum(sdf_prob * self.class_Ids[None, ...], dim=-1, keepdim=True)  # Tensor(N, 1)
#         sdf = (sdf / self.max_class_Id - 0.5) * 2  # to real truncated SDF value, [-1, 1], Tensor(N, 1)
#
#         outputs = torch.cat([rgb, sdf, sdf_entropy, sdf_prob], dim=-1)  # Tensor(N, 10) (10=3+1+1+5)
#         return outputs



# @brief: MLP with SDF classification (pure MLP)
class MLP_reg2(nn.Module):
    def __init__(self,
                 cfg,
                 input_ch=3,
                 input_ch_pos=12,
                 n_hidden=128,
                 n_hidden_rgb=128,
                 n_hidden_sdf=128,
                 n_hidden_branch=128,
                 n_class=5,
                 beta=1.):
        super(MLP_reg2, self).__init__()
        self.cfg = cfg
        self.input_ch = input_ch  # dim of input parametric encoding
        self.input_ch_pos = input_ch_pos + 3  # dim of input positional encoding
        self.n_hidden = n_hidden  # dim of hidden layers, default: 64
        self.n_hidden_rgb = n_hidden_rgb  # dim of RGB embedding, default: 32
        self.n_hidden_sdf = n_hidden_sdf  # dim of SDF embedding, default: 32
        self.n_hidden_branch = n_hidden_branch
        self.beta = beta

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
            nn.Linear(self.n_hidden_rgb + self.input_ch_pos, self.n_hidden_branch),
            nn.ReLU(),
            nn.Linear(self.n_hidden_branch, 3),

            # nn.Linear(self.n_hidden_rgb + self.input_ch_pos, 3)
        )

        # second stage - SDF branch
        self.sdf_linear = nn.Sequential(
            nn.Linear(self.n_hidden_sdf, self.n_hidden_branch),
            nn.ReLU(),
            nn.Linear(self.n_hidden_branch, self.n_class),

            # nn.Linear(self.n_hidden_sdf, self.n_class),
            # nn.Softmax(dim=-1)
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
        # h2 = torch.cat([sdf_embedding, embed_pos_w], dim=-1)
        sdf_prob = self.sdf_linear(sdf_embedding)  # normalized classification probability, Tensor(N, class_num)
        sdf_prob = torch.exp(sdf_prob * self.beta) / torch.sum(torch.exp(sdf_prob * self.beta), dim=-1, keepdim=True)  # beta-softmax

        sdf_entropy = -1. * torch.sum(sdf_prob * torch.log2(sdf_prob + 1e-5), dim=-1, keepdim=True)  # Tensor(N, 1)

        # pred probability --> SDF value
        sdf = torch.sum(sdf_prob * self.class_Ids[None, ...], dim=-1, keepdim=True)  # Tensor(N, 1)
        sdf = ( sdf / self.max_class_Id - 0.5 ) * 2  # to real truncated SDF value, [-1, 1], Tensor(N, 1)

        outputs = torch.cat([rgb, sdf, sdf_entropy, sdf_prob], dim=-1)  # Tensor(N, 10) (10=3+1+1+5)
        return outputs