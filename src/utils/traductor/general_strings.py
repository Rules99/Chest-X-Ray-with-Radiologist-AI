
input_option_q = "Choose an X-Ray image to detect anomalies of the chest (the file must be jpg or png)"
or_im_output = 'Original Image'
def possible_pat(label:str = 'normal'):
    return 'Attention scale'

def pats(label:str)->list:
    weak = f'Negligible {label}'
    mid  = f'Possible {label}'
    imp = f'Critical {label}'
    est = f'{label} detected'
    return [weak,mid,imp,est]
