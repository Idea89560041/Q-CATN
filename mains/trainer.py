from models.networks import ResAdaInGen
import torch
import torch.nn as nn

class dwi_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(dwi_Trainer, self).__init__()
        if hyperparameters['gen']['g_type'] == 'resnet':
            self.gen_a = ResAdaInGen(hyperparameters['input_dim'], hyperparameters['output_dim'], hyperparameters['gen'])
        else:
            raise NotImplementedError

        gpu_ids = hyperparameters['gpu_ids']
        if isinstance(gpu_ids, list) and len(gpu_ids) > 1:
            self.gen_a = torch.nn.DataParallel(self.gen_a, device_ids=gpu_ids)
            self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
        print('Deploy to {}'.format(gpu_ids))
        self.gen_a.to(self.device)

    def forward(self, b0, bvec):
        self.eval()
        x_img = self.gen_a.forward(b0, bvec)
        self.train()
        return x_img