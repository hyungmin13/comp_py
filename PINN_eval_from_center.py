#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
import h5py
from tqdm import tqdm
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data

#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    args = parser.parse_args()
    checkpoint_fol = args.checkpoint
    print(checkpoint_fol, type(checkpoint_fol))

    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)

    a['data_init_kwargs']['path'] = 'DNS/'
    a['problem_init_kwargs']['path_s'] = 'Ground/'
    a['problem_init_kwargs']['problem_name'] = 'TBL'
    
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
        pickle.dump(a,f)
    f.close()
    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
#%% train data의 center로부터의 거리에 따라 Error를 표시 - bins 의 크기에 따라 달라지게 해놓았다.
    output_shape = (213,141,61)
    total_spatial_error = []
    t_pos_un = np.concatenate([train_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i]
                                 for i in range(4)],1).reshape(-1,4)

    t_pos_c = t_pos_un - np.array(all_params["domain"]["in_max"]/2).reshape(-1,4)
    t_pos_c = np.sqrt(t_pos_c[:,1]**2+t_pos_c[:,2]**2+t_pos_c[:,3]**2)
    t_pos_un = t_pos_un.reshape(-1,4)

    counts, bins, bars = plt.hist(t_pos_c, bins=50)

    train_indexes = []
    for i in range(bins.shape[0]-1):
        index = np.where((t_pos_c<bins[i+1])&(t_pos_c>=bins[i]))
        train_indexes.append(index[0])

    t_vel_s_l = []
    t_pos_s_l = []
    for i in range(len(train_indexes)):
        t_vel_s_l.append(train_data['vel'][train_indexes[i],:])
        t_pos_s_l.append(train_data['pos'][train_indexes[i],:])

    t_p_list = []
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    for i in tqdm(range(len(t_pos_s_l)),desc="Train data estimation"):
        p_list = []
        for j in range(t_pos_s_l[i].shape[0]//10000+1):
            pred = model_fn(all_params, t_pos_s_l[i][10000*j:10000*(j+1),:])
            p_list.append(pred)
        p_list = np.concatenate(p_list,0)
        p_u = np.concatenate([p_list[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
        p_u[:,-1] = 1.185*p_u[:,-1]
        t_p_list.append(p_u)

    t_vel_e_l = []
    for i in range(len(t_p_list)):
        t_vel_e_l.append(np.sqrt((np.sqrt(t_p_list[i][:,0]**2+t_p_list[i][:,1]**2+t_p_list[i][:,2]**2)-
                                       np.sqrt(t_vel_s_l[i][:,0]**2+t_vel_s_l[i][:,1]**2+t_vel_s_l[i][:,2]**2))**2)/
                                       np.sqrt(t_vel_s_l[i][:,0]**2+t_vel_s_l[i][:,1]**2+t_vel_s_l[i][:,2]**2))
    
    dist = getattr(st,"norm")
    t_vel_mean_e = []
    for i in range(len(t_vel_e_l)):
        mean_std = dist.fit(t_vel_e_l[i])
        t_vel_mean_e.append(mean_std[0])
    t_vel_mean_e = np.array(t_vel_mean_e)
#%% valid data의 center로부터의 거리에 따라 Error를 표시
    v_pos_un = np.concatenate([valid_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i] 
                                for i in range(4)],1).reshape((-1,)+output_shape+(4,))
    v_pos_c = v_pos_un[1,:,:,:,:].reshape(-1,4) - np.array([all_params["domain"]["in_max"].reshape(1,4)/2])
    v_pos_c = np.sqrt(v_pos_c[:,1]**2+v_pos_c[:,2]**2+v_pos_c[:,3]**2)
    v_pos_un = v_pos_un.reshape(-1,4)
    idx = v_pos_c.argsort(0)
    d_c, counts = np.unique(v_pos_c,return_counts=True)

    v_shape = valid_data['pos'].shape[0]
    outs = []
    for i in tqdm(range(v_shape//10000+1)):
        out = model_fn(all_params, valid_data['pos'][10000*i:10000*(i+1),:])
        outs.append(out)
    outs = np.concatenate(outs)
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    out_u = np.concatenate([all_params["data"][keys[i]]*outs[:,i:(i+1)] for i in range(len(keys))],1)

    v_e_l = []
    p_e_l = []
    for i in range(51):
        v_e = np.sqrt((np.sqrt(out_u[output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),0:1]**2
                               +out_u[output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),1:2]**2
                               +out_u[output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3]**2)
                       -np.sqrt(valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),0:1]**2
                                +valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),1:2]**2
                                +valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3]**2))**2)/np.sqrt(valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),0:1]**2
                                                                                               +valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),1:2]**2
                                                                                               +valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3]**2)
        p_e = np.sqrt((out_u[output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3]
                       -valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3])**2/valid_data['vel'][output_shape[0]*output_shape[1]*output_shape[2]*i:output_shape[0]*output_shape[1]*output_shape[2]*(i+1),2:3]**2)
        v_e_l.append(v_e[idx])
        p_e_l.append(p_e[idx])
    v_e_l = np.hstack(v_e_l)
    p_e_l = np.hstack(p_e_l)

    v_e_t_mean = []
    p_e_t_mean = []
    cnt = 0
    for i in range(len(counts)):
        v_e_t_mean.append(np.mean(v_e_l[cnt:cnt+counts[i],:]))
        p_e_t_mean.append(np.mean(p_e_l[cnt:cnt+counts[i],:]))
        cnt = cnt+counts[i]
    v_e_mean = []
    p_e_mean = []
    for i in range(v_e_l.shape[1]):
        cnt = 0
        v_mean = []
        p_mean = []
        for j in range(len(counts)):
            v_mean.append(np.mean(v_e_l[cnt:cnt+counts[j],:]))
            p_mean.append(np.mean(p_e_l[cnt:cnt+counts[j],:]))
            cnt = cnt+counts[i]
        v_e_mean.append(np.vstack(v_mean))
        p_e_mean.append(np.vstack(p_mean))
    v_e_mean = np.hstack(v_e_mean)
    p_e_mean = np.hstack(p_e_mean)

    e_t_mean = {"vel":v_e_t_mean, "pre":p_e_t_mean}
    e_mean = {"vel":v_e_mean, "pre":p_e_mean}

    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)
    with open("datas/"+checkpoint_fol+"/train_error_from_center.pkl","wb") as f:
        pickle.dump(t_vel_mean_e,f)
    f.close()
    with open("datas/"+checkpoint_fol+"/valid_error_from_center_total.pkl","wb") as f:
        pickle.dump(e_t_mean,f)
    f.close()
    with open("datas/"+checkpoint_fol+"/valid_error_from_center.pkl","wb") as f:
        pickle.dump(e_mean,f)
    f.close()
