from .hrnet import HRNet

def state_dict_back_compat(state_dict):
    # Change this function as necessary to ensure backwards compatibility
    # between earlier commits. Sometimes this will not be possible.
    for k in list(state_dict.keys()):
        if not k.startswith('hrnet.last_layer'):
            continue
        sep_keys = k.split('.')
        sep_keys[1] = 'final_layer'
        new_k = '.'.join(sep_keys)
        state_dict[new_k] = state_dict[k]
        del state_dict[k]