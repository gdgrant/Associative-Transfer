import numpy as np
import os, sys
import pickle
from parameters_sequence import *
import model_sequence as model

weights_file = './savedir/general_model_weights.pkl'
var_dict = pickle.load(open(weights_file, 'rb'))
update_weights(var_dict)

try:
	if len(sys.argv) > 1:
		model.main(sys.argv[1])
	else:
		model.main()
except KeyboardInterrupt:
	quit('Quit by KeyboardInterrupt.')
