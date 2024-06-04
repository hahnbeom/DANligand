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
        self.w_screen_ranking = 0.0
        self.trim_receptor_embedding = True # prv False
        self.max_epoch = 500
        self.debug = False
        self.datasetf = 'data/PLmix.60k.screen.txt'
        self.n_lig_feat = 19
        self.n_lig_emb = 4 # concatenated w/  other m embeddings
        self.struct_loss = 'mse'
        self.input_features = 'base'
        self.pert = False
        self.ligand_model = 'se3'
        self.load_cross = False
        self.cross_eval_struct = False
        self.cross_grid = 0.0
        self.nonnative_struct_weight = 0.2
        self.randomize_grid = 0.0
        self.shared_trigon = True
        self.normalize_Xform = False
        self.lig_to_key_attn = False
        self.keyatomf = 'keyatom.def.npz'

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

    def set_dropout_rate(self, value):
        self.params_grid['dropout_rate'] = value
        self.params_ligand['dropout_rate'] = value
        self.params_TR['dropout_rate'] = value
            
#========================
#---------------------------------
args_XligS = Argument( 0.2, m=64 ) #m: pre-attention channel
args_XligS.params_grid['num_layers_grid'] = 3 #5
args_XligS.params_TR['num_layers_lig'] = 3 #3
args_XligS.params_TR['n_trigon_lig_layers'] = 1 #2
args_XligS.params_TR['n_trigon_key_layers'] = 2 #3
args_XligS.params_ligand['num_layers'] = 4
args_XligS.params_TR['c'] = 64
args_XligS.classification_mode = "former_contrast"
args_XligS.ligand_model = 'gat'
args_XligS.pert = False
args_XligS.feattype('graph')
args_XligS.datasetf = ['data/v4.ext.noChemBl.train.txt','data/v4.BL2.valid.txt']

args_XligS.wTR = 0.1 # weights all structural losses (Ts, Tr)
args_XligS.w_screen = 1e-4
args_XligS.w_screen_contrast = 1e-4
args_XligS.w_screen_ranking = 1e-4

args_XligS.load_cross = True
args_XligS.cross_eval_struct = True
args_XligS.cross_grid = 0.0
args_XligS.nonnative_struct_weight = 0.2

args_XligS.modelname = 'XligS'
args_XligS.shared_trigon = False # originally True
args_XligS.normalize_Xform = False
args_XligS.lig_to_key_attn = False

args_XligSN = copy.deepcopy(args_XligS)
args_XligSN.modelname = 'XligSN'
args_XligSN.shared_trigon = False # originally True
args_XligSN.normalize_Xform = True
args_XligSN.lig_to_key_attn = False

args_XligSAN = copy.deepcopy(args_XligS)
args_XligSAN.modelname = 'XligSAN'
args_XligSAN.shared_trigon = False # originally True
args_XligSAN.normalize_Xform = True
args_XligSAN.lig_to_key_attn = True

args_XallSAN = copy.deepcopy(args_XligS)
args_XallSAN.modelname = 'XallSAN'
args_XallSAN.shared_trigon = False # originally True
args_XallSAN.normalize_Xform = True
args_XallSAN.lig_to_key_attn = True
args_XallSAN.cross_grid = 1.0

args_XligSANt = copy.deepcopy(args_XligSAN)
args_XligSANt.modelname = 'XligSANt'
args_XligSANt.w_screen = 1.0
args_XligSANt.w_screen_contrast = 1.0
args_XligSANt.w_screen_ranking = 15.0

args_Xall04 = copy.deepcopy(args_XallSAN)
args_Xall04.modelname = 'Xall04'
args_Xall04.cross_grid = 0.4

args_L1 = copy.deepcopy(args_XligSAN)
args_L1.modelname = 'L1'
args_L1.params_grid['num_layers_grid'] = 5
args_L1.params_TR['num_layers_lig'] = 3 
args_L1.params_TR['n_trigon_lig_layers'] = 2
args_L1.params_TR['n_trigon_key_layers'] = 3

### w: grid 1.0*{grid-contrast 2.0}, TR 0.1*{mse 1.0 +spread 5.0}

args_L1all04 = copy.deepcopy(args_XligSAN)
args_L1all04.modelname = 'L1all04'
args_L1all04.cross_grid = 0.4
args_L1all04.datasetf = ['data/filt.ext.noChemBl.train.txt','data/v4.BL2.valid.txt']
args_L1all04.params_grid['num_layers_grid'] = 5
args_L1all04.params_TR['num_layers_lig'] = 3 
args_L1all04.params_TR['n_trigon_lig_layers'] = 2
args_L1all04.params_TR['n_trigon_key_layers'] = 3
args_L1all04.w_screen = 1e-4
args_L1all04.w_screen_contrast = 1e-4
args_L1all04.w_screen_ranking = 1e-4

args_mix1 = copy.deepcopy(args_L1all04)
args_mix1.cross_grid = 1.0
args_mix1.modelname = 'mix1'

args_self05t = copy.deepcopy(args_L1all04)
args_self05t.modelname = "self05t"
args_self05t.datasetf = ['data/filt.ext.train.self05.txt','data/v4.BL2.valid.txt']
args_self05t.wGrid = 0.25
args_self05t.w_screen = 0.5
args_self05t.w_screen_contrast = 0.5
args_self05t.w_screen_ranking = 5.0

args_Chemblbal = copy.deepcopy(args_self05t)
args_Chemblbal.modelname = "Chemblbal"
args_Chemblbal.datasetf = ['data/bal.ext.train.self05.txt','data/bal.BL2.valid.txt']

args_Dkey = copy.deepcopy(args_Chemblbal)
args_Dkey.modelname = "Dkey"
args_Dkey.pert = True #conformer input 

args_func = copy.deepcopy(args_mix1)
args_func.modelname = 'func'
args_func.datasetf = ['data/filt.ext.train.self05.txt','data/v4.BL2.valid.txt']
args_func.keyatomf = 'funcatom.def.npz'
args_func.wGrid = 0.5

