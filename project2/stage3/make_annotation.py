import pandas as pd
import os
from PIL import Image

os.chdir('/Users/chenxiahang/Documents/pythondata/project1')

PHASE = ['train', 'val']
SPECIES = ['rabbits', 'rats', 'chickens']  # [0,1,2]
CLASS = ['Mammals', 'Birds']

DATA_info = {'train': {'path': [], 'species': [],'class':[]},
             'val': {'path': [], 'species': [],'class':[]}
             }

for p in PHASE:
    for s in SPECIES:
        DATA_DIR =  p + '/' + s
        DATA_NAME = os.listdir(DATA_DIR)
        DATA_NAME.sort()
        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                if s == 'rabbits':
                    DATA_info[p]['species'].append(0)
                    DATA_info[p]['class'].append(0)
                elif s == 'rats':
                    DATA_info[p]['species'].append(1)
                    DATA_info[p]['class'].append(0)
                else:
                    DATA_info[p]['species'].append(2)
                    DATA_info[p]['class'].append(1)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv('Species_%s_annotation.csv' % p)
    print('Species_%s_annotation file is saved.' % p)


