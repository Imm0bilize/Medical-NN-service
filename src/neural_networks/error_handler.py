def incorrect_start_sess_param():
    raise KeyError(f'Incorrect start session parameter')

def models_weights_isnt_defined():
    raise ValueError('Model weights could not be loaded')