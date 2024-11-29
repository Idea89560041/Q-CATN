import sys
sys.path.append('..')
sys.path.append('.')
import torch
import yaml
import h5py
import time
import numpy as np
from torch.autograd import Variable

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def read_h5_new(h5_file, return_dwi=False):
    gp = h5py.File(h5_file, 'r')
    print(gp.keys())
    b0_ = np.flip(gp['b0'][()].transpose(2, 1, 0), axis=1)
    bvec_ = gp['bvec'][()]
    bval_ = gp['bval'][()]
    bval_max = bval_.max()
    print('bval max: {}'.format(bval_max))
    bval_ /= bval_max
    t1_ = np.flip(gp['t1'][()].transpose(2, 1, 0), axis=1)
    t2_ = np.flip(gp['t2'][()].transpose(2, 1, 0), axis=1)
    t2_[t2_ <= 0.] = 1e-9
    t1_[t1_ <= 0.] = 1e-9
    t1_ /= np.max(t1_, (1, 2))[:, None, None]
    t2_ /= np.max(t2_, (1, 2))[:, None, None]
    t1_[b0_ == 0.] = 0.
    t2_[b0_ == 0.] = 0.

    return_data = {'b0': b0_,
                   't1': t1_,
                   't2': t2_,
                   'bvec': bvec_,
                   'bval': bval_}
    if return_dwi:
        s = time.time()
        dwi_ = np.flip(gp['dwi'][()].transpose(0, 3, 2, 1), axis=2)
        e = time.time()
        print('%.4f sec' % (e - s), dwi_.shape)
    else:
        dwi_ = None
    return_data['dwi'] = dwi_
    return return_data

def synthesize_slice(b0, t2, t1, s, net, bvector, bvalue, pad_x, pad_y, min_x, max_x, min_y, max_y):
    with torch.no_grad():
        _, x, y = b0.shape
        out2d = np.zeros([x, y])
        in_c = 0
        in_slice = None
        if b0 is not None:
            b0_slice = b0[s, min_y:max_y, min_x:max_x]
            in_c += 1
            in_slice = np.expand_dims(b0_slice, -1).astype(np.float32)
        if t2 is not None:
            in_c += 1
            t2[t2 < 0.] = 0.
            t2_slice = t2[s, min_y:max_y, min_x:max_x]
            t2_slice = np.expand_dims(t2_slice, -1).astype(np.float32)
            if in_slice is None:
                in_slice = t2_slice
            else:
                in_slice = np.concatenate((in_slice, t2_slice), axis=-1)
        if t1 is not None:
            in_c += 1
            t1[t1 < 0.] = 0.
            t1_slice = t1[s, min_y:max_y, min_x:max_x]
            t1_slice = np.expand_dims(t1_slice, -1).astype(np.float32)
            if in_slice is None:
                in_slice = t1_slice
            else:
                in_slice = np.concatenate((in_slice, t1_slice), axis=-1)

        raw_x, raw_y = in_slice.shape[0], in_slice.shape[1]
        xs = np.zeros([pad_x, pad_y, in_c])
        start_x = (pad_x - raw_x) // 2
        start_y = (pad_y - raw_y) // 2
        xs[start_x:start_x + raw_x, start_y:start_y + raw_y, :] = in_slice
        xs = xs.transpose(2, 0, 1)

        input = Variable(torch.from_numpy(xs.copy())).unsqueeze(0).cuda().float()
        bvec = Variable(torch.from_numpy(bvector.copy())).unsqueeze(0).cuda().float()
        bval = Variable(torch.from_numpy(bvalue.copy())).unsqueeze(0).cuda().float()
        cond = torch.cat((bvec, bval), dim=-1)
        output = net.forward(input, cond)
        mask = (input[:, 0, :, :] > 0.).float().expand_as(output)
        output = output * mask
        out_np = output[0, 0, start_x:start_x + raw_x, start_y:start_y + raw_y].cpu().numpy()
        out2d[min_y:max_y, min_x:max_x] = out_np

        return out2d