

### load data
read_cur_grid = True
save_grid_path = 'cov_grid.npy'
save_cor_grid_path = 'cor_grid.npy'


### data processing
mean = [0.5, 0.5, 0.5]
sd = [0.5, 0.5, 0.5]
padding = 0
padding_type = 'zero'
random_crop = None
use_float64 = True

### training
lr = 2e-4