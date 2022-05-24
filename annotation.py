#<< --- define custom labels. 
_N = '_N'  # N - normal class - Non-Ectopic
_S = '_S'  # S - SupraVentricular Ectopic Beats (SVEBs)
_V = '_V'  # V - Ventricular Ectopic Beats (VEBs)
_F = '_F'  # F - Fusion Beats
_Q = '_Q'  # Q - Unrecognized
_X = '_X'  # X - Unmapped - if you don't want to use a lable, map it under this class - it should not be used in classification

custom_ants = [ _N, _S, _V, _F, _Q ]  # as recomended by AAMI

# define a mapping dictionary #----------------------------------
custom_cols = { _N:'tab:green',
                _S:'tab:red',
                _V:'tab:blue',
                _F:'tab:purple',
                _Q:'yellow',
                _X:'tab:gray'}
custom_mapping = {
                    #<--- Normal Class
                    'N': _N, # Normal beat
                    'L': _N, # Left bundle branch block beat
                    'R': _N, # Right bundle branch block beat
                    'B': _N, # Bundle branch block beat (unspecified)
    
                    #<--- SVEB
                    'A': _S, # Atrial premature beat
                    'a': _S, # Aberrated atrial premature beat
                    'J': _S, # Nodal (junctional) premature beat
                    'S': _S, # Supraventricular premature or ectopic beat (atrial or nodal)
    
                    #<--- VEB
                    'V': _V, # Premature ventricular contraction
                    'r': _V, # R-on-T premature ventricular contraction
    
                    #<--- FUSION
                    'F': _F, # Fusion of ventricular and normal beat
    
                    #<---* Supraventricular escape - aami says its normal but should be mapped to _S
                    'e': _S, # Atrial escape beat
                    'j': _S, # Nodal (junctional) escape beat
                    'n': _S, # Supraventricular escape beat (atrial or nodal)
    
                    #<---* Ventricular escape - aami says its normal but should be mapped to _V
                    'E': _V, # Ventricular escape beat
    
                    #<--- Paced beats are unmapped - dont use record containing paced beats (mitdb - 102,104,107,217)
                    'f': _X, # Fusion of paced and normal beat
                    '/': _X, # Paced beat
    
                    #<--- Unrecognised or Unclassifiable
                    'Q': _Q, # Unclassifiable 
                    '?': _Q, # Beat not classified during learning

                }
