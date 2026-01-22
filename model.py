import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertConfig, BertTokenizer, BertModel
from torchvision.models import alexnet, resnet18, resnet50, inception_v3, vgg19
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import pickle
import math
import clip

import numpy as np
import torch as th
import torch.nn as nn
import helpers


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class _Fusion(nn.Module):
    def __init__(self,  input_sizes):
        """
        Base class for the fusion module

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.output_size = None

    def forward(self, inputs):
        raise NotImplementedError()

    @classmethod
    def get_weighted_sum_output_size(cls, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        assert all(s == flat_sizes[0] for s in flat_sizes), f"Fusion method {cls.__name__} requires the flat output" \
                                                            f" shape from all backbones to be identical." \
                                                            f" Got sizes: {input_sizes} -> {flat_sizes}."
        return [flat_sizes[0]]

    def get_weights(self, softmax=True):
        out = []
        if hasattr(self, "weights"):
            out = self.weights
            if softmax:
                out = nn.functional.softmax(self.weights, dim=-1)
        return out

    def update_weights(self, inputs, a):
        pass


class Mean(_Fusion):
    def __init__(self,  input_sizes):
        """
        Mean fusion.

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__( input_sizes)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        return th.mean(th.stack(inputs, -1), dim=-1)


class WeightedMean(_Fusion):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self,  input_sizes):
        super().__init__(input_sizes)
        self.weights = nn.Parameter(th.full((2,), 1 / 2), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = nn.functional.softmax(weights, dim=0)
    out = th.sum(weights[None, None, :] * th.stack(tensors, dim=-1), dim=-1)
    return out


MODULES = {
    "mean": Mean,
    "weighted_mean": WeightedMean,
}


def get_fusion_module(input_sizes):
    return WeightedMean(input_sizes)

class DDC(nn.Module):
    def __init__(self, input_dim,n_hidden,n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()

        hidden_layers = [nn.Linear(input_dim, n_hidden), nn.ReLU()]
        use_bn=True
        if use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, mindum_dim=2048, out_dim=1024, dropout_prob=0.1):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=768, mindum_dim=4096, out_dim=1024, dropout_prob=0.1):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out




# class model(nn.Module):
#     def __init__(self, img_dim=4096, text_dim=1000, mid_dim_i=2048, mid_dim_t=2048, feature_dim=1024):
#         super(model, self).__init__()
#
#         self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim_i, out_dim=feature_dim)
#         self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim_t, out_dim=feature_dim)
#
#
#         self.feat_dim = feature_dim
#
#         self.fusion_dim=[[1024],[1024]]
#         self.fusion = get_fusion_module(self.fusion_dim)
#
#         self.ddc=DDC(input_dim=1024,n_hidden=512,n_clusters=10)
#
#         from loss import Loss
#         self.loss=Loss(funcs="ddc_1|ddc_2|ddc_3|contrast")
#         self.apply(helpers.he_init_weights)
#
#
#         #self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())
#
#
#
#     def forward(self, img, text):
#
#         img_features = self.imgnn(img)
#         img_features = l2norm(img_features, dim=1)
#
#         text_features = self.textnn(text)
#         text_features = l2norm(text_features, dim=1)
#
#         self.backbone_outputs=[img_features,text_features]
#         self.fused=self.fusion(self.backbone_outputs)
#         self.projections = torch.cat(self.backbone_outputs, dim=0)
#
#         self.output, self.hidden = self.ddc(self.fused)
#
#         return img_features,text_features, self.output

# class Encoder(nn.Module):
#     def __init__(self, input_dim, feature_dim=1024):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, feature_dim),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# class Decoder(nn.Module):
#     def __init__(self, input_dim, feature_dim=1024):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(feature_dim, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, input_dim)
#         )
#
#     def forward(self, x):
#         return self.decoder(x)

# class Network(nn.Module):
#     def __init__(self, view, input_size, feature_dim=1024, high_feature_dim=512):
#
#
#         super(Network, self).__init__()
#         self.encoders = []
#         #self.decoders = []
#         for v in range(view):
#             self.encoders.append(Encoder(input_size[v], feature_dim))
#             #self.decoders.append(Decoder(input_size[v], feature_dim))
#         self.encoders = nn.ModuleList(self.encoders)
#         #self.decoders = nn.ModuleList(self.decoders)
#
#         self.fusion_dim=[[1024],[1024]]
#         self.fusion = get_fusion_module(self.fusion_dim)
#         self.projector = nn.Identity()
#         self.ddc=DDC(input_dim=1024,n_hidden=512,n_clusters=20)
#         from loss import Loss
#         self.loss=Loss(funcs="ddc_1|ddc_2|ddc_3|contrast")
#
#
#         #self.image_head = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(),nn.Linear(1024, 512))
#         #self.text_head = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(),nn.Linear(1024, 512))
#
#         self.apply(helpers.he_init_weights)
#
#     def forward(self, img, text):
#
#         view1_feature=self.encoders[0](img)
#         view2_feature = self.encoders[1](text)
#
#
#         norm_view1_feature = torch.norm(view1_feature, dim=1, keepdim=True)
#         norm_view2_feature = torch.norm(view2_feature, dim=1, keepdim=True)
#         norm_view1_feature= view1_feature / norm_view1_feature
#         norm_view2_feature = view2_feature / norm_view2_feature
#
#         self.backbone_outputs=[view1_feature,view2_feature]
#         self.fused=self.fusion(self.backbone_outputs)
#         self.projections = self.projector(torch.cat(self.backbone_outputs, dim=0))
#         self.output, self.hidden = self.ddc(self.fused)
#
#         #img_c=self.image_head(view1_feature)
#         #text_c=self.text_head(view2_feature)
#
#         #view1_feature_rec=self.decoders[0](view1_feature)
#         #view2_feature_rec = self.decoders[1](view2_feature)
#
#         #return norm_view1_feature,norm_view2_feature, view1_feature,view2_feature,view1_feature_rec,view2_feature_rec,self.output
#         return norm_view1_feature,norm_view2_feature, self.output
#
#     def forward_mse(self, xs):
#         xrs = []
#         for v in range(self.view):
#             z = self.encoders[v](xs[v])
#             xrs.append(self.decoders[v](z))
#
#         return xrs
# class AttentionLayer(nn.Module):
#     def __init__(self, cfg, input_size):
#         """
#         EAMC attention net
#
#         :param cfg: Attention config
#         :type cfg: config.eamc.defaults.AttentionLayer
#         :param input_size: Input size
#         :type input_size: Union[List[int, ...], Tuple[int, ...], ...]
#         """
#         super().__init__()
#         self.tau = cfg.tau
#
#         self.mlp = MLP(cfg.mlp_config, input_size=[input_size[0] * cfg.n_views])
#         self.output_layer = nn.Linear(self.mlp.output_size[0], cfg.n_views, bias=True)
#         self.weights = None
#
#     def forward(self, xs):
#         h = th.cat(xs, dim=1)
#         act = self.output_layer(self.mlp(h))
#         e = nn.functional.softmax(th.sigmoid(act) / self.tau, dim=1)
#         # e = nn.functional.softmax(act, dim=1)
#         self.weights = th.mean(e, dim=0)
#         return self.weights


#    Image to Text: MAP: 0.5455
# Text to Image: MAP: 0.5558

# class Model(nn.Module):
#     def __init__(self, img_input_dim=4096, text_input_dim=768, common_emb_dim=1024):
#         super(Model, self).__init__()
#
#         self.img_net = nn.Sequential(
#             nn.Linear(img_input_dim, common_emb_dim*2),
#             nn.ReLU(),
#
#             #nn.Linear(common_emb_dim,256),
#             #nn.ReLU(),
#             #nn.Linear(256, 128),
#
#             #nn.GELU(),
#             nn.Dropout(0.5)
#         )
#         # self.img_net = nn.Sequential(
#         #     nn.Linear(img_input_dim, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear(common_emb_dim*2,output_dim),
#         # )
#         # self.text_net = nn.Sequential(
#         #     nn.Linear(txt_input_dim, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear(common_emb_dim*2,output_dim),
#         # )
#
#         self.text_net = nn.Sequential(
#             nn.Linear(text_input_dim, common_emb_dim*2),
#             nn.ReLU(),
#             #nn.Linear(common_emb_dim,256),
#             #nn.ReLU(),
#             #nn.Linear(256, 128),
#             #nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         #self.l1 = nn.Linear(common_emb_dim* 2, common_emb_dim)
#         self.drop_out = nn.Dropout(0.5)
#         self.fusion_dim=[[128],[128]]
#
#         self.fusion = get_fusion_module(self.fusion_dim)
#         #self.projector = nn.Identity()
#         #if True:
#             #self._initialize_weights()
#         self.apply(helpers.he_init_weights)
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Linear):
#     #             nn.init.xavier_uniform_(m.weight)
#
#     def forward(self, img, text):
#
#
#         view1_feature = self.img_net(img.float())
#         view2_feature = self.text_net(text.float())
#         #view1_feature = self.drop_out(F.relu(self.l1(view1_feature)))
#
#         #view1_feature = self.drop_out(F.gelu(self.l1(view1_feature)))
#         #view2_feature = self.drop_out(F.gelu(self.l1(view2_feature)))
#         #view1_feature = l2norm(view1_feature, dim=1)
#         #view2_feature = l2norm(view2_feature, dim=1)
#
#         self.backbone_outputs=[view1_feature,view2_feature]
#         fused=self.fusion(self.backbone_outputs)
#         self.projections = torch.cat(self.backbone_outputs, dim=0)
#
#         return view1_feature,view2_feature, fused

#Image to Text: MAP: 0.5710
#Text to Image: MAP: 0.5580
# class Model(nn.Module):
#     def __init__(self, img_input_dim=4096, text_input_dim=768, common_emb_dim=1024):
#         super(Model, self).__init__()
#
#         self.img_net = nn.Sequential(
#             nn.Linear(img_input_dim, common_emb_dim*2),
#             nn.ReLU(),
#
#             #nn.Linear(common_emb_dim,256),
#             #nn.ReLU(),
#             #nn.Linear(256, 128),
#
#             #nn.GELU(),
#             nn.Dropout(0.5)
#         )
#         # self.img_net = nn.Sequential(
#         #     nn.Linear(img_input_dim, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear(common_emb_dim*2,output_dim),
#         # )
#         # self.text_net = nn.Sequential(
#         #     nn.Linear(txt_input_dim, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
#         #     nn.RELU(),
#         #     nn.Linear(common_emb_dim*2,output_dim),
#         # )
#
#         self.text_net = nn.Sequential(
#             nn.Linear(text_input_dim, common_emb_dim*2),
#             nn.ReLU(),
#             #nn.Linear(common_emb_dim,256),
#             #nn.ReLU(),
#             #nn.Linear(256, 128),
#             #nn.GELU(),
#             nn.Dropout(0.5)
#         )
#
#         self.l1 = nn.Linear(common_emb_dim* 2, common_emb_dim)
#         self.drop_out = nn.Dropout(0.5)
#         self.fusion_dim=[[128],[128]]
#
#         self.fusion = get_fusion_module(self.fusion_dim)
#         #self.projector = nn.Identity()
#         #if True:
#             #self._initialize_weights()
#         self.apply(helpers.he_init_weights)
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Linear):
#     #             nn.init.xavier_uniform_(m.weight)
#
#     def forward(self, img, text):
#
#
#         view1_feature = self.img_net(img.float())
#         view2_feature = self.text_net(text.float())
#         view1_feature = self.drop_out(F.relu(self.l1(view1_feature)))
#
#         #view1_feature = self.drop_out(F.gelu(self.l1(view1_feature)))
#         view2_feature = self.drop_out(F.gelu(self.l1(view2_feature)))
#         #view1_feature = l2norm(view1_feature, dim=1)
#         #view2_feature = l2norm(view2_feature, dim=1)
#
#         self.backbone_outputs=[view1_feature,view2_feature]
#         fused=self.fusion(self.backbone_outputs)
#         self.projections = torch.cat(self.backbone_outputs, dim=0)
#
#         return view1_feature,view2_feature, fused

class Model(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=768, common_emb_dim=1024):
        super(Model, self).__init__()

        self.img_net = nn.Sequential(
            nn.Linear(img_input_dim, common_emb_dim*2),
            nn.ReLU(),

            #nn.Linear(common_emb_dim,256),
            #nn.ReLU(),
            #nn.Linear(256, 128),

            #nn.GELU(),
            nn.Dropout(0.5)
        )
        # self.img_net = nn.Sequential(
        #     nn.Linear(img_input_dim, common_emb_dim*2),
        #     nn.RELU(),
        #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
        #     nn.RELU(),
        #     nn.Linear(common_emb_dim*2,output_dim),
        # )
        # self.text_net = nn.Sequential(
        #     nn.Linear(txt_input_dim, common_emb_dim*2),
        #     nn.RELU(),
        #     nn.Linear( common_emb_dim*2, common_emb_dim*2),
        #     nn.RELU(),
        #     nn.Linear(common_emb_dim*2,output_dim),
        # )

        self.text_net = nn.Sequential(
            nn.Linear(text_input_dim, common_emb_dim*2),
            nn.ReLU(),
            #nn.Linear(common_emb_dim,256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.GELU(),
            nn.Dropout(0.5)
        )

        self.l1 = nn.Linear(common_emb_dim* 2, common_emb_dim)
        self.drop_out = nn.Dropout(0.5)
        self.fusion_dim=[[128],[128]]

        self.fusion = get_fusion_module(self.fusion_dim)
        #self.projector = nn.Identity()
        #if True:
            #self._initialize_weights()
        self.apply(helpers.he_init_weights)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)

    def forward(self, img, text):


        view1_feature = self.img_net(img.float())
        view2_feature = self.text_net(text.float())
        view1_feature = self.drop_out(F.relu(self.l1(view1_feature)))

        #view1_feature = self.drop_out(F.gelu(self.l1(view1_feature)))
        view2_feature = self.drop_out(F.gelu(self.l1(view2_feature)))
        #view1_feature = l2norm(view1_feature, dim=1)
        #view2_feature = l2norm(view2_feature, dim=1)

        self.backbone_outputs=[view1_feature,view2_feature]
        fused=self.fusion(self.backbone_outputs)
        #self.projections = torch.cat(self.backbone_outputs, dim=0)

        return view1_feature,view2_feature, fused


class model_2(nn.Module):
    def __init__(self, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024):
        super(model_2, self).__init__()

        self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)

        #self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)  # 不考虑bias，权重归一化之后就是Proxy-NCA, Normlized Softmax
        #self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)
        # if init_weight:
        #     self.__init_weight()

    # def __init_weight(self):
    #     nn.init.kaiming_normal_(self.centers, mode='fan_out')
    #     nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, img, text):
        # the normalization of class proxies, the more obvious its effect is in the higher-dimensional representation space
        #self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
        #self.centers.data = l2norm(self.centers.data, dim=0)

        img_features = self.imgnn(img.float())
        img_features = l2norm(img_features, dim=1)
        #img_pred = self.predictLayer(img_features)

        text_features = self.textnn(text.float())
        text_features = l2norm(text_features, dim=1)
        #text_pred = self.predictLayer(text_features)

        return  img_features, text_features

class Model_3(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=768, common_emb_dim=1024):
        super(Model_3, self).__init__()

        self.img_net = nn.Sequential(
            nn.Linear(img_input_dim, common_emb_dim*2),
            nn.ReLU(),
           nn.Dropout(0.5),
            nn.Linear(common_emb_dim * 2, common_emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.text_net = nn.Sequential(
            nn.Linear(text_input_dim, common_emb_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(common_emb_dim * 2, common_emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.l1 = nn.Linear(common_emb_dim* 2, 1024)
        self.drop_out = nn.Dropout(0.5)

        self.apply(helpers.he_init_weights)



    def forward(self, img, text):


        view1_feature = self.img_net(img.float())
        view2_feature = self.text_net(text.float())
        view1_feature = self.drop_out(F.relu(self.l1(view1_feature)))
        view2_feature = self.drop_out(F.gelu(self.l1(view2_feature)))

        norm_view1_feature = torch.norm(view1_feature, dim=1, keepdim=True)
        norm_view2_feature = torch.norm(view2_feature, dim=1, keepdim=True)
        view1_feature = view1_feature / norm_view1_feature
        view2_feature = view2_feature / norm_view2_feature

        return view1_feature,view2_feature

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=1024, mindum_dim=2048, out_dim=1024, dropout_prob=0.1):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=1024, mindum_dim=2048, out_dim=1024, dropout_prob=0.1):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out

class model_SDCML(nn.Module):
    def __init__(self, num_class, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024, init_weight=True):
        super(model_SDCML, self).__init__()

        self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)

        self.n_classes = num_class
        self.feat_dim = feature_dim
        self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)  # 不考虑bias，权重归一化之后就是Proxy-NCA, Normlized Softmax
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)

        #self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.feat_dim * 2, nhead=1,dim_feedforward=256)

        if init_weight:
            self.__init_weight()



    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, img, text):
        # the normalization of class proxies, the more obvious its effect is in the higher-dimensional representation space
        self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
        self.centers.data = l2norm(self.centers.data, dim=0)

        img_features = self.imgnn(img.float())
        img_features = l2norm(img_features, dim=1)
        img_pred = self.predictLayer(img_features)

        text_features = self.textnn(text.float())
        text_features = l2norm(text_features, dim=1)
        text_pred = self.predictLayer(text_features)

        return self.centers, img_features, text_features, img_pred, text_pred

class FusionNet(nn.Module):
    def __init__(self, num_class):
        super(FusionNet, self).__init__()

        self.n_classes = num_class

        self.pred = nn.Sequential(
            nn.Linear(1024*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_classes)
        )



    # pt1, img1, centers1, corners1, normals1, neighbor_index1
    def forward(self,  img_feature,text_feature):

        # print(pt_base.size(), mesh_base.size())
        concatenate_feature = torch.cat([img_feature,text_feature], dim = 1)

        # print(concatenate_feature.size())
        fused_pred = self.pred(concatenate_feature)

        return fused_pred

class SDMCL(nn.Module):
    def __init__(self,num_class):
        super(SDMCL, self).__init__()

        self.n_classes=num_class
        self.network=model_SDCML(self.n_classes)
        self.fusion_pre=FusionNet(self.n_classes)

    def forward(self,  img_feature,text_feature):
        centers, img_features, text_features, img_pred, text_pred=self.network(img_feature,text_feature)

        fused_pred = self.fusion_pre(img_features,text_features)

        return centers, img_features, text_features, img_pred, text_pred,fused_pred






# class Model_3(nn.Module):
#     def __init__(self, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024):
#         super(Model_3, self).__init__()
#
#         self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
#         self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)
#
#         self.l1 = nn.Linear(feature_dim, feature_dim)
#         self.drop_out = nn.Dropout(0.1)
#
#         self.apply(helpers.he_init_weights)
#
#
#
#     def forward(self, img, text):
#
#
#         view1_feature = self.img_net(img.float())
#         view2_feature = self.text_net(text.float())
#         view1_feature = l2norm(self.drop_out(F.gelu(self.l1(view1_feature))))
#         view2_feature = self.drop_out(F.gelu(self.l1(view2_feature)))
#
#
#
#         # norm_view1_feature = torch.norm(view1_feature, dim=1, keepdim=True)
#         # norm_view2_feature = torch.norm(view2_feature, dim=1, keepdim=True)
#         # view1_feature = view1_feature / norm_view1_feature
#         # view2_feature = view2_feature / norm_view2_feature
#
#         return view1_feature,view2_feature


if __name__ == '__main__':


    from main import load_dataset
    batch_size =128
    dataset_name='nus_wide'
    dataloader=load_dataset(dataset_name,batch_size)
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    M = model().to(device)
    for img, text,label in train_loader:
        text = text.float().to(device)
        img = img.float().to(device)

        I,T,C=M(img,text)
        print('pass')