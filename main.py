from train_stage0 import *
from train_stage1 import *
from train_stage2 import *
from train_stage3 import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

name = 'model_dir'

train0(nb_epoch=50,
          batch_size=12,
          store_name=name,
          resume=False,
          start_epoch=0,
          model_path='')

train1(nb_epoch=50,
          batch_size=6,
          store_name=name,
          resume=False,
          start_epoch=0,
          model_path='')

train2(nb_epoch=50,
          batch_size=6,
          store_name=name,
          resume=False,
          start_epoch=0,
          model_path='')

train3(nb_epoch=50,
          batch_size=4,
          store_name=name,
          resume=False,
          start_epoch=0,
          model_path='')
