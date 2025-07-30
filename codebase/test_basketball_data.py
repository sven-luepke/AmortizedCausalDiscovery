#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from utils import data_loader
import numpy as np

class Args:
    def __init__(self):
        self.suffix = '_basketball10'
        self.batch_size_multiGPU = 1
        self.datadir = 'data'
        self.training_samples = 0
        self.test_samples = 0

def test_basketball_data_loading():
    args = Args()
    
    try:
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = data_loader.load_data(args)
        print('Basketball data loaded successfully!')
        print(f'Train batches: {len(train_loader)}')
        print(f'Valid batches: {len(valid_loader)}')  
        print(f'Test batches: {len(test_loader)}')
        print(f'Loc range: [{loc_min:.2f}, {loc_max:.2f}]')
        print(f'Vel range: [{vel_min:.2f}, {vel_max:.2f}]')
        
        # Test a batch
        for batch in train_loader:
            print(f'Batch data shape: {batch[0].shape}')
            print(f'Batch edges shape: {batch[1].shape}')
            break
            
        return True
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basketball_data_loading() 