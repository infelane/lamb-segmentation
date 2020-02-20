modalities_list = ['clean', 'rgb', 'ir', 'irr', 'xray', 'uvf']

modalities_set_names = ['clean', 'all', '5']


def get_mod_set(mod_name):
    
    _modality_exist(mod_name)
    
    if mod_name == 'all':
        return modalities_list[:]
    elif mod_name == 'clean':
        return modalities_list[0]
    else:
        try:
            if int(mod_name) == 5:
                return modalities_list[:5]
            else: NotImplementedError()
        except ValueError as verr:
            pass
            NotImplementedError()
        
        NotImplementedError(f'{mod_name}')


def get_mod_n_in(mod_name):
    _modality_exist(mod_name)
    
    if mod_name == 'all':
        return 12
    elif mod_name == 'clean':
        return 3
    else:
        try:
            if int(mod_name) == 5:
                return 9
            else:
                NotImplementedError(f'{mod_name}')
        except ValueError as verr:
            pass
            NotImplementedError(f'{mod_name}')
        
        NotImplementedError(f'{mod_name}')


def _modality_exist(mod):
    
    assert str(mod) in modalities_set_names, (mod, modalities_set_names)
