import copy

class train_arguments:
    def __init__(self,
                 modelname,
                 topk,
                 neighmode,
                 LR,
                 pert,
                 mixkey,
                 w_reg,
                 w_spread,
                 w_class,
                 classification_mode,
                 datapath='/ml/motifnet/TRnet.ligand/grid1.0/', #'/home/j2ho/DB/Motifnet'
                 K=4,
                 maxepoch=300,
                 max_subset=5,
                 datasetf='data/0519_decoy.list'
    ):
        self.modelname =           modelname
        self.datapath =            datapath
        self.topk =                topk          
        self.neighmode =           neighmode
        self.LR =                  LR
        self.pert =                pert             
        self.mixkey =              mixkey           
        self.w_reg =               w_reg           
        self.w_spread =            w_spread
        self.w_class =             w_class
        self.classification_mode = classification_mode
        self.maxepoch =            maxepoch
        self.K =                   K
        self.max_subset =          max_subset
        self.datasetf =            datasetf

        # ddp related
        self.world_size = 1
        self.rank = 0
        self.gpu = 0

args_trsf = train_arguments(modelname='trsf0',
                            topk= 16,
                            neighmode='topk',
                            LR=1.0e-4,
                            pert= False,
                            mixkey=True,
                            w_reg=1.e-4,
                            w_spread=3.0,
                            w_class=5.0,
                            classification_mode='tank')

args_up = copy.deepcopy(args_trsf)
args_up.modelname = "trsf0_upweight"
args_up.w_class = 20.0
args_up.classification_mode = 'ligand'

args_mix = train_arguments(modelname='trsf0_mix',
                          topk= 16,
                          neighmode='topk',
                          LR=1.0e-4,
                          pert= False,
                          mixkey=True,
                          w_reg=1.e-4,
                          w_spread=3.0,
                          w_class=30.0,
                          classification_mode='ligand')

args_scratch = copy.deepcopy(args_up)
args_scratch.modelname = 'scratch'

args_dynamic2 = copy.deepcopy(args_up)
args_dynamic2.modelname = 'trsf_dynamic' #dynamic-discrimination
args_dynamic2.K = -1

args_grid15 = copy.deepcopy(args_dynamic2)
args_grid15.modelname = 'trsf_grid1.5' #dynamic-discrimination
args_grid15.datapath = '/ml/motifnet/TRnet.combo/'
#args_grid15.K = -1
# 1:3:20

args_slower15 = copy.deepcopy(args_grid15)
args_slower15.modelname = 'trsf_slower1.5' #dynamic-discrimination
args_slower15.LR = 2.5e-5

args_slower15b = copy.deepcopy(args_slower15)
args_slower15b.modelname = 'trsf_slower1.5b'
args_slower15b.datasetf = 'data/0620_decoy.list'

args_trsf2 = copy.deepcopy(args_grid15)
args_trsf2.modelname = 'trsf2'
args_trsf2.datasetf = 'data/0620_decoy.list'
args_trsf2.LR = 1.0e-4
#args_trsf2.w_reg = 1.0e-3 # stronger
args_trsf2.max_subset = 4
args_trsf2.w_class = 100.0

args_grid15_scratch = copy.deepcopy(args_grid15)
args_grid15_scratch.modelname = 'trsf_grid1.5_set' #dynamic-discrimination
args_grid15_scratch.datasetf = 'data/0620_decoy.list'

args_freeze = copy.deepcopy(args_grid15)
args_freeze.modelname = 'trsf1_freeze',
args_freeze.datasetf = 'data/0620_decoy.list'
args_freeze.datapath = '/ml/motifnet/TRnet.combo/'

