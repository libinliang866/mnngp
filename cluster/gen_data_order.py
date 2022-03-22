import numpy as np

def gen_order(train_size, val_size, test_size, total_train, total_test, n_sample, order = [0, 0]):
    
    trains = np.zeros((n_sample, train_size))
    vals = np.zeros((n_sample, val_size))
    tests = np.zeros((n_sample, test_size))
    rng = np.random.default_rng()
    if order == None:
        for i in np.arange(n_sample):
            idx_train = rng.choice(total_train, size=(train_size + val_size), replace = False)
            vals[i, ] = idx_train[train_size:]
            trains[i, ] = idx_train[:train_size]
            tests[i, ] = rng.choice(total_test, size = test_size, replace = False)
                     
    else:
        start, step_size = order
        idx = np.linspace(0, total_train, num = total_train, endpoint = False, dtype=np.int32) 
        for i in np.arange(n_sample):
            start = start % train_size
            if (start + train_size > total_train):
                idx_val = idx[(start + train_size - total_train) : start] 
                trains[i, ] = np.delete(idx, idx_val)
                
            else:
                idx_train = np.linspace(start, start + train_size, num = train_size, endpoint = False, dtype=np.int32)
                trains[i, ] = idx_train
                idx_val = np.delete(idx, idx_train)
            vals[i, ] = idx_val[:val_size]
            tests[i, ] = rng.choice(total_test, size = test_size, replace = False) 
            start += step_size
    return {'trains_idx' : trains, 'vals_idx' : vals, 'tests_idx' : tests}
    
    
if __name__ == '__main__':
    import os
    import sys
    import pickle as pkl
    if len(sys. argv) != 4:
        raise TypeError('Please provide the directory of source code, that of output, and ouput code!!!')
    else:
        input_file = open(sys.argv[1],'r')
        pars = list(input_file.read().split())
        train_size = int(pars[0])
        val_size = int(pars[1])
        test_size = int(pars[2])
        total_train = int(pars[3])
        total_test = int(pars[4])
        n_sample = int(pars[5])
        if len(pars) == 7:
            order = None
        else:
            order = [int(pars[6]), int(pars[7])]
        out_dir = sys.argv[2]
        out_code = int(sys.argv[3])
    output = gen_order(train_size, val_size, test_size, total_train, total_test, n_sample, order)
    path = '{}/order_{}.pkl'.format(out_dir, out_code)
    file = open(path, 'wb')
    pkl.dump(output, file)
    file.close() 
    
    