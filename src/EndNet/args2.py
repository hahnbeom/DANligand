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
        self.classification_mode = "ligand_v2"
        self.LR = 1.0e-4
        self.wTR = 0.2
        self.wGrid = 1.0
        self.w_reg = 1e-10
        self.w_contrast = 2.0 # divided by ngrid
        self.w_false = 0.2 #1.0/6.0 # 1.0/ngroups null contribution
        self.w_spread = 5.0
        self.w_screen = 0.0
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

args_base = Argument( 0.1 )
args_base.modelname = 'base' 

args_cat = Argument( 0.1 )
args_cat.modelname = 'catonly'
args_cat.wTR = 0.0

args_deep = Argument( 0.1, m=64 ) #m: pre-attention channel
args_deep.modelname = 'deep'
args_deep.params_grid['num_layers_grid'] = 5
args_deep.params_TR['num_layers_lig'] = 3
args_deep.params_TR['n_trigonometry_module_stack'] = 5
args_deep.wTR = 0.2

args_addcls = copy.deepcopy(args_deep)
args_addcls.modelname = 'addcls'
args_addcls.w_screen = 30.0

args_ldeep = copy.deepcopy(args_deep)
args_ldeep.modelname = 'ldeep'
args_ldeep.params_TR['num_layers_lig'] = 5

args_laddcls = Argument( 0.2, m=64 ) #m: pre-attention channel
args_laddcls.modelname = 'laddcls'
args_laddcls.params_grid['num_layers_grid'] = 5
args_laddcls.params_TR['num_layers_lig'] = 5
args_laddcls.params_TR['n_trigonometry_module_stack'] = 5
args_laddcls.wTR = 0.2
args_laddcls.w_screen = 30.0
args_laddcls.classification_mode = 'ligand_v3'

args_monitor = Argument( 0.2, m=64 ) #m: pre-attention channel
args_monitor.modelname = 'monitor'
args_monitor.params_grid['num_layers_grid'] = 5
args_monitor.params_TR['num_layers_lig'] = 5
args_monitor.params_TR['n_trigonometry_module_stack'] = 5
args_monitor.classification_mode = 'ligand_v3'

args_combo = copy.deepcopy(args_laddcls)
args_combo.modelname = 'combo1'
args_combo.classification_mode = 'combo_v1'

#----------------------------------------------------
args_monitor = Argument( 0.2, m=64 ) #m: pre-attention channel
args_monitor.modelname = 'monitor'
args_monitor.datasetf = 'data/AddChembl.120k.screen.txt'
args_monitor.params_grid['num_layers_grid'] = 5
args_monitor.params_TR['num_layers_lig'] = 3
args_monitor.params_TR['n_trigonometry_module_stack'] = 5
args_monitor.classification_mode = 'ligand_v3'
args_monitor.wTR = 0.2
args_monitor.w_screen = 30.0

args_former = Argument( 0.2, m=64 ) #m: pre-attention channel
args_former.modelname = 'former'
args_former.datasetf = 'data/AddChembl.120k.screen.txt'
args_former.params_grid['num_layers_grid'] = 5
args_former.params_TR['num_layers_lig'] = 3
args_former.params_TR['n_trigon_lig_layers'] = 2
args_former.params_TR['n_trigon_key_layers'] = 3
args_former.params_TR['c'] = 64
args_former.classification_mode = 'former'
args_former.wTR = 0.2
args_former.w_screen = 30.0

args_formerNR = copy.deepcopy(args_former)
args_formerNR.modelname = 'former_nrb'
args_formerNR.datasetf = 'data/AddChembl.NR.txt'

args_formerNR_scratch = copy.deepcopy(args_formerNR)
args_formerNR_scratch.modelname = 'former_nrbS'

args_ex1 = copy.deepcopy(args_formerNR)
args_ex1.modelname = 'former_ex1'
args_ex1.datasetf = ['data/AddChembl.NR2.train.txt','data/AddChembl.NR2.valid.txt']
args_ex1.feattype("ex1")

args_ex1T = copy.deepcopy(args_ex1)
args_ex1T.modelname = 'former_ex1T'
args_ex1T.LR = 1.0e-6
args_ex1T.datasetf = ['data/ChemblOnly.NR2.train.txt','data/AddChembl.NR2.valid.txt'] #'data/ChemblOnly.NR2.valid.txt']

args_ex1rotT = copy.deepcopy(args_ex1T)
args_ex1rotT.modelname = 'former_ex1rotT'
#args_ex1rotT.LR = 1.0e-5

args_ex1rot = copy.deepcopy(args_ex1)
args_ex1rot.modelname = 'former_ex1rot'
args_ex1rot.pert = True

#========================
args_cross = copy.deepcopy(args_ex1rot)
args_cross.modelname = 'former_cross'
args_cross.datasetf = ['data/v3.train.txt','data/v3.valid.txt'] #'data/ChemblOnly.NR2.valid.txt']
args_cross.wTR = 0.1
args_cross.w_screen = 10.0 # removed 0.2 front so effectively 50.0 in previous unit

args_crossT = copy.deepcopy(args_cross)
args_crossT.modelname = 'former_crossT'
args_crossT.datasetf = ['data/v3.trainChemblOnly.txt','data/v3.validChemblOnly.txt'] 
args_crossT.LR = 1.0e-6

