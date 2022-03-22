
import numpy as np
  
def gen_param(bias, bias_sub, weights, weights_sub, q, q_sub, depth, depth_sub, eps, eps_sub):
    bias_var = np.linspace(bias[0], bias[1], num=bias[2], dtype = np.float64)
    weights_var = np.linspace(weights[0], weights[1], num=weights[2], dtype = np.float64)
    qs = np.linspace(q[0], q[1], num=q[2], dtype = np.int32)
    depths = np.linspace(depth[0], depth[1], num=depth[2], dtype = np.int32)
    epses = np.linspace(eps[0], eps[1], num=eps[2], dtype = np.float64)
    if bias_sub != None:
        bias_idx = np.linspace(bias_sub[0], bias_sub[1], num = bias_sub[2], dtype = np.int32)
        bias_var = bias_var[bias_idx]
    if weights_sub != None:
        weights_idx = np.linspace(weights_sub[0], weights_sub[1], num = weights_sub[2], dtype = np.int32)
        weights_var = weights_var[weights_idx]
    if q_sub != None:
        q_idx = np.linspace(q_sub[0], q_sub[1], num = q_sub[2], dtype = np.int32)
        qs = qs[q_idx] 
    if depth_sub != None:
        depth_idx = np.linspace(depth_sub[0], depth_sub[1], num = depth_sub[2], dtype = np.int32)
        depths = depths[depth_idx] 
    if eps_sub != None:
        eps_idx = np.linspace(eps_sub[0], eps_sub[1], num = eps_sub[2], dtype = np.int32)
        epses = epses[eps_idx] 
    total = len(qs) * len(depths) * len(bias_var) * len(weights_var) * len(epses) 

    pars = np.zeros((total, 5))
    count = 0

    for q in qs:
        for depth in depths:
          for sig_b in bias_var:
            for sig_w in weights_var:
                for ep in epses:
                    pars[count, ] = [q, depth, sig_b, sig_w, np.exp(ep)]
                    count += 1
    return pars
  
  
if __name__ == '__main__': 

    import sys
    import pickle as pkl
    if len(sys. argv) != 4:
        raise TypeError('Please provide the directory of source code, that of output, and ouput code!!!')
    else:
        input_file = open(sys.argv[1],'r')
        pars = input_file.readlines()
        
        bias = [float(x) for x in pars[0].strip().split()]
        bias[2] = int(bias[2])
        
        if pars[1].strip() == 'None':
            bias_sub = None
        else:
            bias_sub = [int(x) for x in pars[1].strip().split()]
        
        weights = [float(x) for x in pars[2].strip().split()]
        weights[2] = int(weights[2])
        
        if pars[3].strip() == 'None':
            weights_sub = None
        else:
            weights_sub = [int(x) for x in pars[3].strip().split()]  

        q = [int(x) for x in pars[4].strip().split()]

        if pars[5].strip() == 'None':
            q_sub = None
        else:
            q_sub = [int(x) for x in pars[5].strip().split()]  
        
        depth = [int(x) for x in pars[6].strip().split()]

        if pars[7].strip() == 'None':
            depth_sub = None
        else:
            depth_sub = [int(x) for x in pars[7].strip().split()]  

        eps = [float(x) for x in pars[8].strip().split()]
        eps[2] = int(eps[2])
        if pars[9].strip() == 'None':
            eps_sub = None
        else:
            eps_sub = [int(x) for x in pars[9].strip().split()]  
        
        precision = int(pars[10].strip().split()[0])
        out_dir = sys.argv[2]
        out_code = int(sys.argv[3])
        

        
        output = gen_param(bias, bias_sub, weights, weights_sub, q, q_sub, depth, depth_sub, eps, eps_sub)
        
        for idx, pars in enumerate(output):
            path = '{}/{}/input.{}'.format(out_dir, out_code, idx)
          
            par = ["%.{}f".format(precision) % par for par in pars]
            out_file = open(path,'w')
            out_file.writelines(' '.join(par))
            out_file.close() 
         

















