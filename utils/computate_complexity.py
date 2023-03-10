# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:07:21 2023

@author: AndreasMiltiadous
"""
from ..machine_learning import dice_models
from ..machine_learning.dice_models import AbstractDualInput
import torch
import warnings

from thop import profile
from thop import clever_format


def computate_model_complexity(model):
    '''
    Calculates the complexity of a DICE-type model

    Parameters
    ----------
    model : model that inherits from AbstractDualInput

    Returns
    -------
    macs : int
    params : int

    '''
    if isinstance(model, AbstractDualInput):
        input1 = torch.randn(32, 30, 5, 19)
        input2 = torch.randn(32, 30, 5, 19)
        macs, params = profile(model, inputs=(input1,input2 ))
        macs, params = clever_format([macs, params], "%.3f")
        print("macs: ", macs)
        print("params: ",params)
    else:
        warnings.warn("Invalid object type - expected AbstractDualInput subclass", Warning)
        return 0,0
    return macs,params

