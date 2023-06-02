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
                 K=4,
                 maxepoch=300
    ):
        self.modelname =           modelname
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

args_freeze = train_arguments(modelname='trsf0_freeze',
                            topk= 16,
                            neighmode='topk',
                            LR=5.0e-5,
                            pert= False,
                            mixkey=True,
                            w_reg=1.e-4,
                            w_spread=3.0,
                            w_class=1.0,
                            classification_mode='ligand')

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
args_dynamic2.modelname = 'dd1' #dynamic-discrimination
args_dynamic2.K = -1


