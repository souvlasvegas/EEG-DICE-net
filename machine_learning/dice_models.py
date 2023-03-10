# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:56:15 2023

@author: AndreasMiltiadous
"""


from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

import torch
import abc



class AbstractDualInput(torch.nn.Module, abc.ABC):
    """
    Abstract class for all DICE models to inherit,
    so as to be in the same hierarchy
    """
    def __init__(self):
        super(AbstractDualInput, self).__init__()
        
    @abc.abstractmethod
    def forward(self, x):
        pass




class Model_early_concat(AbstractDualInput):
    """
    
    Concatenation of Convolution layers, and then Transformer.
    """
    def __init__(self):
        super(Model_early_concat, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        #self.batchnorm1 = torch.nn.BatchNorm1d(19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        #self.batchnorm2 = torch.nn.BatchNorm1d(19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding = Summer(PositionalEncoding1D(38))
        ################# Transformer Enconder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=38, nhead=2)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=38)
        self.output = torch.nn.Linear(in_features=38, out_features=16)
        self.batchnorm3=torch.nn.BatchNorm1d(16)
        #ADD DROPOUT LAYERS
        self.dropout1 = torch.nn.Dropout(0.20)
        
        #antreas
        self.output2=torch.nn.Linear(in_features=16,out_features=1)
        
        #ADD DROPOUT LAYERS
        self.dropout2 = torch.nn.Dropout(0.20)


    def forward(self, input1, input2):
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1
        #batchnorm
        depthwise_conv_output1 = torch.nn.functional.relu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2
        #batchnorm
        depthwise_conv_output2 = torch.nn.functional.relu(depthwise_conv_output2)
        concat_1_2 = torch.cat((depthwise_conv_output1, depthwise_conv_output2), dim=1)
        concat_1_2 = concat_1_2.permute(0,2,1)                                           # proper format (dimensions)
        #print(concat_1_2.size())
        ############################################################################
        positional_enc = self.positional_encoding(concat_1_2)  # positional encoding
        ############################################################################
        cls_token = torch.randn((positional_enc.shape[0], 1, positional_enc.shape[-1]))  # randomly initialize cls token
        tokens = torch.column_stack((cls_token, positional_enc))                         # tokens is of shape [B, 1+T, F]    
        transformer_output_all = self.transformer_encoder(tokens)                # Transformer
        transformer_output_1 = transformer_output_all[:,0,:]                             # Take the cls token output
        ############################################################################
        layer_norm_output = self.layernorm(transformer_output_1)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm3(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output) ## dropout layer 2
        output2=self.output2(output)                                          # linear output
        return output2
        ############################################################################


class Model_cls(AbstractDualInput):
    def __init__(self):
        super(Model_cls, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding = Summer(PositionalEncoding1D(38))
        ######################CLS TOKEN NEW
        self.class_token = torch.nn.Parameter(torch.randn(1, 26, 1))
        ################# Transformer Enconder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=39, nhead=3)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=26)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=26, out_features=16)
        self.batchnorm1=torch.nn.BatchNorm1d(16)
        #self.output2=torch.nn.Linear(in_features=16,out_features=8)
        self.dropout2=torch.nn.Dropout(0.20)
        #self.batchnorm2=torch.nn.BatchNorm1d(8)
        self.output3=torch.nn.Linear(in_features=16,out_features=1)
        self.dropout3 = torch.nn.Dropout(0.20)


    def forward(self, input1, input2):
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        concat_1_2 = torch.cat((depthwise_conv_output1, depthwise_conv_output2), dim=1)
    
        concat_1_2 = concat_1_2.permute(0,2,1)                                           # proper format (dimensions)
        ############################################################################
        positional_enc = self.positional_encoding(concat_1_2)  # positional encoding
        ############################################################################
        transformer_output_all=torch.cat([self.class_token.expand(input1.shape[0],-1,-1),positional_enc],dim=2)
        transformer_output_all = self.transformer_encoder(transformer_output_all)                # Transformer
        transformer_output_1 = transformer_output_all[:,:,0]                             # Take the cls token output
        ############################################################################
        layer_norm_output = self.layernorm(transformer_output_1)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output)

        output3=self.output3(output)                                          # linear output
        return output3
        ############################################################################


class Model_cls_late_concat(AbstractDualInput):
    def __init__(self):
        super(Model_cls_late_concat, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))
        ######################CLS TOKEN NEW
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 26, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 26, 1))
        ################# Transformer Enconder Layer
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=52)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=52, out_features=24)
        self.batchnorm1=torch.nn.BatchNorm1d(24)
        #self.output2=torch.nn.Linear(in_features=16,out_features=8)
        self.dropout2=torch.nn.Dropout(0.20)
        #self.batchnorm2=torch.nn.BatchNorm1d(8)
        self.output3=torch.nn.Linear(in_features=24,out_features=1)
        self.dropout3 = torch.nn.Dropout(0.20)


    def forward(self, input1, input2):
        #fdsafgs
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        
        ###permute conv1 and conv2
        depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)
        depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        positional_enc1=self.positional_encoding1(depthwise_conv_output1)
        positional_enc2=self.positional_encoding2(depthwise_conv_output2)
        transformer_output_all1=torch.cat((self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1),dim=2)
        transformer_output_all1 = self.transformer_encoder1(transformer_output_all1)
        transformer_output_1 = transformer_output_all1[:,:,0]
        transformer_output_all2=torch.cat((self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2),dim=2)
        transformer_output_all2 = self.transformer_encoder2(transformer_output_all2)
        transformer_output_2 = transformer_output_all2[:,:,0] 
        concat_1_2 = torch.cat((transformer_output_1, transformer_output_2), dim=1)
        ############################################################################
        layer_norm_output = self.layernorm(concat_1_2)
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output)
        output3=self.output3(output)                                       # linear output
        return output3
        ############################################################################

class Model_mean_cls_late_concat(AbstractDualInput):
    def __init__(self):
        super(Model_mean_cls_late_concat, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))
        ######################CLS TOKEN NEW
    
        ################# Transformer Enconder Layer
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=52)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=52, out_features=24)
        self.batchnorm1=torch.nn.BatchNorm1d(24)
        #antreas
        #self.output2=torch.nn.Linear(in_features=16,out_features=8)
        self.dropout2=torch.nn.Dropout(0.20)
        #self.batchnorm2=torch.nn.BatchNorm1d(8)
        self.output3=torch.nn.Linear(in_features=24,out_features=1)
        self.dropout3 = torch.nn.Dropout(0.20)
    

    def forward(self, input1, input2):
        # print('input1', input1.shape)
        # print('input2', input2.shape)
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        
        ###permute conv1 and conv2
        
        depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)
        depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        positional_enc1=self.positional_encoding1(depthwise_conv_output1)
        positional_enc2=self.positional_encoding2(depthwise_conv_output2)
        
        cls_token1=torch.mean(positional_enc1,dim=2).unsqueeze(-1)
        cls_token2=torch.mean(positional_enc2,dim=2).unsqueeze(-1)
    
        transformer_output_all1=torch.cat((cls_token1,positional_enc1),dim=2)
        transformer_output_all1 = self.transformer_encoder1(transformer_output_all1)
        transformer_output_1 = transformer_output_all1[:,:,0] 
        
        transformer_output_all2=torch.cat((cls_token2,positional_enc2),dim=2)
        transformer_output_all2 = self.transformer_encoder2(transformer_output_all2)
        transformer_output_2 = transformer_output_all2[:,:,0] 
        
        concat_1_2 = torch.cat((transformer_output_1, transformer_output_2), dim=1)
        ############################################################################
        layer_norm_output = self.layernorm(concat_1_2)
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output)
        output3=self.output3(output)                                          # linear output
        return output3
        ############################################################################


class Model_all_tokens(AbstractDualInput):
    def __init__(self):
        super(Model_all_tokens, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding = Summer(PositionalEncoding1D(38))
        ######################CLS TOKEN NEW
        self.class_token = torch.nn.Parameter(torch.randn(1, 26, 1))
        ################# Transformer Enconder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=39, nhead=3)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        ################# Output Layer
        
        self.n_hidden=200
        
        self.layernorm = torch.nn.LayerNorm(normalized_shape=26*39)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=26*39, out_features=self.n_hidden)
        self.batchnorm1=torch.nn.BatchNorm1d(self.n_hidden)
        #antreas
        #self.output2=torch.nn.Linear(in_features=16,out_features=8)
        self.dropout2=torch.nn.Dropout(0.20)
        #self.batchnorm2=torch.nn.BatchNorm1d(8)
        self.output3=torch.nn.Linear(in_features=self.n_hidden,out_features=1)
        self.dropout3 = torch.nn.Dropout(0.20)



    def forward(self, input1, input2):
        # print('input1', input1.shape)
        # print('input2', input2.shape)
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        concat_1_2 = torch.cat((depthwise_conv_output1, depthwise_conv_output2), dim=1)
    
        concat_1_2 = concat_1_2.permute(0,2,1)                                           # proper format (dimensions)
        
        ############################################################################
        positional_enc = self.positional_encoding(concat_1_2)  # positional encoding
        ############################################################################
        transformer_output_all=torch.cat([self.class_token.expand(input1.shape[0],-1,-1),positional_enc],dim=2)
        transformer_output_all = self.transformer_encoder(transformer_output_all)                # Transformer
        ############################################################################
        x=transformer_output_all.reshape(-1,26*39)
        layer_norm_output = self.layernorm(x)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output)

        output3=self.output3(output)                                          # linear output
        return output3
        ############################################################################
    
    
class Model_no_encoder(AbstractDualInput):
    def __init__(self):
        super(Model_no_encoder, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)

        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)

        ################# Positional Encoding (Sinsoidal)
        
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=38*26)
        self.output = torch.nn.Linear(in_features=38*26, out_features=16)
        self.batchnorm3=torch.nn.BatchNorm1d(16)
        #ADD DROPOUT LAYERS
        self.dropout1 = torch.nn.Dropout(0.20)
        
        #antreas
        self.output2=torch.nn.Linear(in_features=16,out_features=1)
        
        #ADD DROPOUT LAYERS
        self.dropout2 = torch.nn.Dropout(0.20)


    def forward(self, input1, input2):
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze()                      # conv1

        depthwise_conv_output1 = torch.nn.functional.relu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze()                      # conv2

        depthwise_conv_output2 = torch.nn.functional.relu(depthwise_conv_output2)
        concat_1_2 = torch.cat((depthwise_conv_output1, depthwise_conv_output2), dim=1)
        concat_1_2 = concat_1_2.permute(0,2,1)                                           # proper format (dimensions)
        x=concat_1_2.reshape(-1,26*38)
        ############################################################################
        layer_norm_output = self.layernorm(x)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm3(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output) ## dropout layer 2
        output2=self.output2(output)                                          # linear output
        return output2
        ############################################################################