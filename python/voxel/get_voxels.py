import os
import numpy as np
import binvox_rw

path = dict()
path['data'] = 'Z:\data\img2mesh'

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

def genlist(cls):
    print("genlist running...")
    paramsdir = os.path.join(path['data'], class2uid[cls], 'obj_models')
    onlyfiles = [f for f in os.listdir(paramsdir) if os.path.isfile(os.path.join(paramsdir, f))]
    onlyfiles.sort()
    num_all = len(onlyfiles)
    fileset = []
    set_file = os.path.join(paramsdir, 'estimated_objs.list')
    with open(set_file, 'w') as tr:
        for i, muid in enumerate(onlyfiles):
            fileset.append(muid)
            tr.write(muid+'\n')
    return fileset
    print("genlist done")

def prepare_data(cls):
    print("prepare_data running...")
    paramsdir = os.path.join(path['data'], class2uid[cls], 'obj_models')
    if not os.path.exists(paramsdir):
        os.makedirs(paramsdir)
    set_file = os.path.join(paramsdir, 'estimated_objs.list')
    if not os.path.isfile(set_file):
        genlist(cls)
    _create_datafile(cls, 'estimated_objs')

def _create_datafile(cls, setname):
    modeldir = os.path.join(path['data'], class2uid[cls], 'obj_models')
    setfile = os.path.join(modeldir, setname+'.list')

    vxls = []
    with open(setfile, 'r') as fp:
        for line in fp:
            muid = line[:-1]
            muid = muid.split('.')[0]
            bvfile = os.path.join(modeldir, muid+'.binvox')
            #if os.path.isfile(bvfile):
                #os.system('rm -rf {}'.format(bvfile))
            objfile = os.path.join(modeldir, muid+'.obj')
            os.system('./binvox -d 30 -cb -e -t binvox -ri {}'.format(objfile))
            #os.system('rm -rf {}'.format(bvfile))
