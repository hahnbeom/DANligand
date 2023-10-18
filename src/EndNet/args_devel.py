import copy

class GridArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers_grid=2,
                 l0_in_features=102,
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

class LigandArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers=2,
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

class TRArgs:
    def __init__(self,
                 dropout_rate=0.1,
                 num_layers_lig=2,
                 num_channels=32,
                 num_degrees=3,
                 n_heads_se3=4,
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
        
        self.params['num_layers_lig']	 = num_layers_lig		
        self.params['num_channels']	 = num_channels	# ligand se3 hidden
        self.params['num_degrees']	 = num_degrees		
        self.params['n_heads_se3']	 = n_heads_se3		
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
        self.w_contrast = 2.0 # divided by ngrid
        self.w_false = 0.2 #1.0/6.0 # 1.0/ngroups null contribution
        self.w_spread = 5.0
        self.w_screen = 0.0
        self.w_screen_contrast = 0.0
        self.trim_receptor_embedding = True # prv False
        self.max_epoch = 200
        self.debug = False
        self.datasetf = 'data/PLmix.60k.screen.txt'
        self.n_lig_emb = 4 # concatenated w/  other m embeddings
        self.input_features = 'base'
        self.pert = False

    def feattype(self, feat):
        if feat == 'base':
            pass
        elif feat == 'ex1':
            self.input_features = feat
            self.params_ligand['l0_in_features'] = 18 #+q, sasa, occl
            self.params_grid['l0_in_features'] = 104 #+q, occl

#========================
args_base = Argument( 0.2, m=64 ) #m: pre-attention channel
args_base.datasetf = ['data/v3.devel.train.txt','data/v3.devel.valid.txt']
args_base.params_grid['num_layers_grid'] = 3 #5
args_base.params_TR['num_layers_lig'] = 3 #3
args_base.params_TR['n_trigon_lig_layers'] = 1 #2
args_base.params_TR['n_trigon_key_layers'] = 2 #3
args_base.params_TR['c'] = 64
args_base.pert = True
args_base.modelname = 'former_base'
args_base.classification_mode = "former_contrast"
args_base.w_screen_contrast = 0.1 #default 0
args_base.wTR = 0.1 
args_base.w_screen = 10.0 # removed 0.2 front so effectively 50.0 in previous unit
#wGrid: cat 1.0, contrast 2.0, false 0.2
