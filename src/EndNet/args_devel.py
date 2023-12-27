import copy

class GridArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers_grid=2,
                 l0_in_features=102,
                 n_heads=4,
                 num_channels=32,
                 num_edge_features=3,
                 l0_out_features=32, 
	         ntypes=6
    ):
        self.params = {}
        self.params['dropout_rate']     = dropout_rate
        
        self.params['num_layers_grid']	= num_layers_grid 
        self.params['l0_in_features']	= l0_in_features
        self.params['num_edge_features']= num_edge_features
        self.params['l0_out_features']	= l0_out_features
        self.params['ntypes']		= ntypes
        self.params['n_heads']	        = n_heads
        self.params['num_channels']     = num_channels

class LigandArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers=2,
                 n_heads=4,
                 num_channels=32,
                 l0_in_features=15,
                 l0_out_features=32, 
                 num_edge_features=5,
    ):
        self.params = {}
        self.params['dropout_rate']     = dropout_rate
        self.params['num_layers']	= num_layers
        self.params['l0_in_features']	= l0_in_features
        self.params['num_edge_features']= num_edge_features
        self.params['l0_out_features']	= l0_out_features
        self.params['n_heads']	        = n_heads		
        self.params['num_channels']     = num_channels

class TRArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers_lig=2,
                 num_channels=32,
                 num_degrees=3,
                 #n_heads_se3=4,
                 div=4,
                 l0_in_features_rec=32, #embedding from GridNet
                 l0_in_features_lig=15, #input
                 l1_in_features=0,
                 l1_out_features=0,
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 m=32, # trigonometry_input_channel
                 c=32, # trigonometry_mid_channel
                 n_trigon_lig_layers = 3,
                 n_trigon_key_layers = 3
    ):
        self.params = {}
        self.params['dropout_rate'] = dropout_rate
        
        #self.params['num_layers_lig']	 = num_layers_lig  unused
        self.params['num_channels']	 = num_channels	# ligand se3 hidden
        self.params['num_degrees']	 = num_degrees		
        #self.params['n_heads_se3']	 = n_heads_se3		
        self.params['div']		 = div			
        self.params['l0_in_features_lig']= l0_in_features_lig		
        self.params['l0_in_features_rec']= l0_in_features_rec		
        self.params['l1_in_features']	 = l1_in_features		
        self.params['l1_out_features']	 = l1_out_features		
        self.params['num_edge_features'] = num_edge_features		
        self.params['l0_out_features_lig'] = m #se3_lig output dim; should be identical
        self.params['d']                 = c
        self.params['m']	         = m #channel1 dim in Triangular
        self.params['c']		 = c #channel2 dim in Triangular
        self.params['n_trigon_lig_layers'] = n_trigon_lig_layers
        self.params['n_trigon_key_layers'] = n_trigon_key_layers

class Argument:
    def __init__(self, dropout_rate=0.1, m=32 ):
        self.dropout_rate = dropout_rate
        self.params_grid = GridArgs(dropout_rate=dropout_rate).params
        self.params_grid['l0_out_features'] = m
        
        self.params_ligand = LigandArgs(dropout_rate=dropout_rate).params
        self.params_ligand['l0_out_features'] = m
        
        self.params_TR   = TRArgs(dropout_rate=dropout_rate).params
        self.params_TR['m'] = m
        self.params_TR['l0_out_features_lig'] = m
        self.classification_mode = "former"
        self.LR = 1.0e-4
        self.wTR = 0.2
        self.wGrid = 1.0
        self.w_reg = 1e-10
        self.screenloss = 'BCE'
        self.w_contrast = 2.0 # divided by ngrid
        self.w_false = 0.2 #1.0/6.0 # 1.0/ngroups null contribution
        self.w_spread = 5.0
        self.w_screen = 0.0
        self.w_screen_contrast = 0.0
        self.trim_receptor_embedding = True # prv False
        self.max_epoch = 200
        self.debug = False
        self.datasetf = 'data/PLmix.60k.screen.txt'
        self.n_lig_feat = 19
        self.n_lig_emb = 4 # concatenated w/  other m embeddings
        self.input_features = 'base'
        self.pert = False
        self.ligand_model = 'se3'

    def feattype(self, feat):
        self.input_features = feat
        if feat == 'base':
            pass
        elif feat == 'ex1':
            self.params_ligand['l0_in_features'] = 18 #+q, sasa, occl
            self.params_grid['l0_in_features'] = 104 #+q, occl
        elif feat == 'ex2':
            self.n_lig_feat = 32 ## CHECK; org + O/N gentypes
            self.params_ligand['l0_in_features'] = 18 #+q, sasa, occl
            #self.params_ligand['num_edge_features'] = 4 # drop 
            self.params_grid['l0_in_features'] = 104 #+q, occl
        elif feat == 'graph':
            self.params_ligand['l0_in_features'] = 18
            #self.n_lig_feat = 19
            #self.n_lig_emb = 8
            #self.params_ligand['num_edge_features'] = 4 # drop distance
            self.params_grid['l0_in_features'] = 104 #+q, occl
        elif feat == 'graphex':
            self.params_ligand['l0_in_features'] = 69 #+q, sasa, occl + (51) 1-hot atomic-FP
            self.n_lig_feat = 19+128 #+ecfp4 N=128
            self.n_lig_emb = 8
            self.params_ligand['num_edge_features'] = 4 # drop distance
            self.params_grid['l0_in_features'] = 104 #+q, occl

            
#========================
args_base = Argument( 0.2, m=64 ) #m: pre-attention channel
args_base.datasetf = ['data/v3.devel.train.txt','data/v3.devel.valid.txt']
args_base.params_grid['num_layers_grid'] = 3 #5
#args_base.params_ligand['num_layers'] = 3 #below doesn't works
args_base.params_TR['num_layers_lig'] = 3 #3
args_base.params_TR['n_trigon_lig_layers'] = 1 #2
args_base.params_TR['n_trigon_key_layers'] = 2 #3
args_base.params_TR['c'] = 64
args_base.pert = False
args_base.modelname = 'base'
args_base.classification_mode = "former_contrast"
args_base.wTR = 0.1 
args_base.w_screen = 10.0 # removed 0.2 front so effectively 50.0 in previous unit
args_base.w_screen_contrast = 1.0 # removed 0.2 front so effectively 50.0 in previous unit
args_base.feattype('ex1')
#wGrid: cat 1.0, contrast 2.0, false 0.2

args_based2 = copy.deepcopy(args_base)
args_based2.modelname = 'based2'
args_based2.datasetf = ['data/v3.BRd2.train.txt','data/v3.BRd2.valid.txt']

args_based2re = copy.deepcopy(args_base)
args_based2re.modelname = 'based2re'
args_based2re.datasetf = ['data/v3.BRd2.train.txt','data/v3.BRd.valid.txt']

args_based3 = copy.deepcopy(args_base)
args_based3.modelname = 'based3'
args_based3.datasetf = ['data/v3.BRd3.train.txt','data/v3.BRd3.valid.txt']

args_based3a = copy.deepcopy(args_base)
args_based3a.modelname = 'based3a'
args_based3a.datasetf = ['data/v3.BRd3a.train.txt','data/v3.BRd3.valid.txt']

args_baseBRDd = copy.deepcopy(args_base)
args_baseBRDd.modelname = 'baseBRDd'
args_baseBRDd.datasetf = ['data/v3.BRDd.train.txt','data/v3.BRd.valid.txt']

args_baseBL = copy.deepcopy(args_base)
args_baseBL.modelname = 'baseBL'
args_baseBL.datasetf = ['data/v3.BL.train.txt','data/v3.BL.valid.txt']

args_baseBLCD = copy.deepcopy(args_base)
args_baseBLCD.modelname = 'baseBLCDDd'
args_baseBLCD.datasetf = ['data/v3.BLCD.train.txt','data/v3.BRd.valid.txt']
args_baseBLCD.pert = True

args_baseBLCDnoDd = copy.deepcopy(args_base)
args_baseBLCDnoDd.modelname = 'baseBLCD'
args_baseBLCDnoDd.datasetf = ['data/v3.BLCDnoDd.train.txt','data/v3.BRd.valid.txt']

args_baseBRCD = copy.deepcopy(args_base)
args_baseBRCD.modelname = 'baseBRCD'
args_baseBRCD.datasetf = ['data/v3.BRCD.train.txt','data/v3.BRd.valid.txt']

args_baseBRDao = copy.deepcopy(args_base)
args_baseBRDao.modelname = 'baseBRDao'
args_baseBRDao.datasetf = ['data/v3.BRDao.train.txt','data/v3.BRd.valid.txt']

args_baseBRDdo = copy.deepcopy(args_base)
args_baseBRDdo.modelname = 'baseBRDdo'
args_baseBRDdo.datasetf = ['data/v3.BRDdo.train.txt','data/v3.BRd.valid.txt']

args_baseBLDd4 = copy.deepcopy(args_base)
args_baseBLDd4.modelname = 'baseBLDd4'
args_baseBLDd4.datasetf = ['data/v3.BLDd4.train.txt','data/v3.BRd.valid.txt']

args_baseBL2 = copy.deepcopy(args_base)
args_baseBL2.modelname = 'baseBL2'
args_baseBL2.datasetf = ['data/v3.BL2.train.txt','data/v3.BL2.valid.txt']

args_baseBL2rank = copy.deepcopy(args_base)
args_baseBL2rank.modelname = 'baseBL2rank'
args_baseBL2rank.datasetf = ['data/v3.BL2.train.txt','data/v3.BL2.valid.txt']
args_baseBL2rank.w_screen = 5.0
args_baseBL2rank.w_screen_contrast = 1.0
args_baseBL2rank.screenloss = 'combo'

args_trim = copy.deepcopy(args_base)
args_trim.modelname = 'baseBL2'
args_trim.datasetf = ['data/v3.BL2.train.txt','data/v3.BL2.valid.txt']

#---------------------------------
args_GAT = copy.deepcopy(args_base)
args_GAT.modelname = 'baseGAT'
args_GAT.datasetf = ['data/v3.devel.train.txt','data/v3.devel.valid.txt']
args_GAT.ligand_model = 'gat'
args_GAT.params_ligand['num_layers'] = 4
#args_GAT.params_TR['num_layers_lig'] = 4 #3
args_GAT.LR = 1.0e-4
args_GAT.w_screen = 0.1
args_GAT.feattype('graph')

args_debug = copy.deepcopy(args_GAT)
args_debug.modelname = 'debug'
args_debug.datasetf = ['data/debug.txt','data/debug.txt']

args_GAT2 = copy.deepcopy(args_GAT)
args_GAT2.modelname = 'baseGAT2'
args_GAT2.datasetf = ['data/v3.BL2.train.txt','data/v3.BL2.valid.txt']
args_GAT2.w_screen = 5.0
