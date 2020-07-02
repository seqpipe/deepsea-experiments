'''
Created on Tue Jun  4 14:16:07 2019

@author: dmarr
'''

import torch

bestloss = 10000


def bestmodel(deepsea, save_model_time, valid_loss):
    global bestloss

    if valid_loss < bestloss:
        bestloss = valid_loss

        torch.save(
            deepsea,
            'model/model{save_model_time}/deepsea_net_bestmodel.pkl'.format(
                save_model_time=save_model_time))

        torch.save(
            deepsea.state_dict(),
            'model/model{save_model_time}/deepsea_net_params_bestmodel.pkl'.
            format(save_model_time=save_model_time))

    return True
