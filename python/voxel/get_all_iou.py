import os
import numpy as np
import scipy.io
import binvox_rw

path = dict()
path['model_gt'] = 'Z:\datasets\FreeFormDeformation'
path['model_hat'] = 'Z:\data\img2mesh'

class2uid = {
    'bottle'        : '02876657',
    'bicycle'       : '02834778',
    'knife'         : '03624134',
    'chair'         : '03001627',
    'car'           : '02958343',
    'diningtable'   : '04379243',
    'sofa'          : '04256520',
    'bed'           : '02818832',
    'dresser'       : '02933112',
    'aeroplane'     : '02691156',
    'motorbike'     : '03790512',
    'bus'           : '02924116',
}

def compute_iou(cls):
    modeldir_gt = os.path.join(path['model_gt'], class2uid[cls], 'rendered')
    modeldir_hat = os.path.join(path['model_hat'], class2uid[cls], 'obj_models')

    setname = 'estimated_objs'
    setfile = os.path.join(modeldir_hat, setname+'.list')

    iou = []
    print('Computing IoU...')
    with open(setfile, 'r') as fp:
        for line in fp:
            muid = line[:-1]
            muid_hat = muid.split('.')[0]
            muid_gt = 'render'+muid_hat[5:]
            bvfile_gt = os.path.join(modeldir_gt, muid_gt+'.binvox')
            bvfile_hat = os.path.join(modeldir_hat, muid_hat+'.binvox')

            with open(bvfile_gt, 'rb') as bvf:
                vxl_gt = binvox_rw.read_as_3d_array(bvf)

            with open(bvfile_hat, 'rb') as bvf:
                vxl_hat = binvox_rw.read_as_3d_array(bvf)

             # The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0
            intersection = np.sum(np.logical_and(vxl_hat.data, vxl_gt.data))
            union = np.sum(np.logical_or(vxl_hat.data, vxl_gt.data))
            IoU = intersection / union
            print(IoU)

            iou.append(IoU)

    filename = os.path.join(modeldir_hat, 'all_iou_test.mat')
    scipy.io.savemat(filename, {'all_iou_test': iou})

    print(np.mean(iou))
    print('Finally done!')
