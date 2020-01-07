modalities_list = ['clean', 'rgb', 'ir', 'irr', 'xray', 'uvf']

modalities_set_names = ['clean', 'all']


def get_mod_set(name):
    
    modality_exist(name)
    
    if name == 'all':
        return modalities_list[:]
    elif name == 'clean':
        return modalities_list[0]
    else:
        NotImplementedError(f'{name}')
        
        
def modality_exist(mod):
    
    assert mod in modalities_set_names
