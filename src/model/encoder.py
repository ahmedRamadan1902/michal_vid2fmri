import timm
import torch
import torch.nn as nn
import numpy as np

import settings
from model.layer import norm_linear

class VidEncoderModel(nn.Module):
    def __init__(self, embed_size=512, backbone_name="resnet18", linear_pool=None, adaptive_pool=1, dropout_rate=0.2):
        super(VidEncoderModel, self).__init__()

        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.linear_pool = linear_pool
        self.adaptive_pool = adaptive_pool

        self.dropout  = nn.Dropout(dropout_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d(adaptive_pool)
        self.max_pool = nn.AdaptiveMaxPool2d(adaptive_pool)
        
        print(f"{backbone_name} ->")
        self.prepare_encoder(backbone_name)

    def get_linear_encoder(self, feature_size, feature_extracted):
        return nn.Sequential(nn.Flatten(2,-1),
                             nn.Dropout(self.dropout_rate),
                             norm_linear(feature_size[1] * feature_size[2], feature_extracted),
                             nn.Flatten(1,-1),
                             )

    def prepare_encoder(self, model_name):
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        with torch.no_grad():
            o = self.backbone(torch.rand(2, 3, settings.IMG_SIZE, settings.IMG_SIZE))

        features_pooled = 2*(self.adaptive_pool**2)
        feature_size_list = np.array([list(oo.size()[1:]) for oo in o])
        print(f"{feature_size_list} -> avg + max pool {self.adaptive_pool} * {self.adaptive_pool} ", end="")

        if self.linear_pool is not None:
            features_pooled += self.linear_pool
            self.linear_pool_layers = nn.ModuleList()
        
            for i, feature_size in enumerate(feature_size_list):
                self.linear_pool_layers.append(self.get_linear_encoder(feature_size, self.linear_pool))

            print(f"+ linear pool [{self.linear_pool}] ", end="")

        embed_input_size = np.sum(feature_size_list[:,0]) * features_pooled
        print(f"-> {embed_input_size} - > {self.embed_size}")
        self.final_embed = norm_linear(embed_input_size, self.embed_size)


    def pool_and_flatten(self, input):
        x = torch.cat([self.avg_pool(input), self.max_pool(input)], dim=1)
        x = x.flatten(1,-1)
        return x

    def extract_features(self, input):
        x_f = []
        # get output from each layer of backbone network
        with torch.no_grad():
            o = self.backbone(input)
            for x in o:
                x_f.append(x)
        
        # pool and encode outputs
        for i, x in enumerate(x_f):
            x_pool = self.pool_and_flatten(x)
            # encode using flattening and linear layers
            if self.linear_pool is not None:
                x_lin = self.linear_pool_layers[i](x)
                x_f[i] = torch.cat([x_lin, x_pool], dim=1)
            else:
                x_f[i] = x_pool

        x_f = torch.cat(x_f, dim=1)
        x_f = self.dropout(x_f)
        x_f = self.final_embed(x_f)
        return x_f

    def forward(self, input):
        B, T, C, X, Y = input.size()
        input = input.reshape(-1, C, X, Y)
        x = self.extract_features(input)
        x = x.reshape(B, T, self.embed_size)
        return x