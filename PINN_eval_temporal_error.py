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

def error_metric(pred, test, div):
    out = np.linalg.norm(pred-test)/np.linalg.norm(div)
    return out

def error_metric2(pred, test):
    f = np.concatenate([(pred[0]-test[0]).reshape(-1,1),
                        (pred[1]-test[1]).reshape(-1,1),
                        (pred[2]-test[2]).reshape(-1,1)],1)
    div = np.concatenate([(test[0]).reshape(-1,1),
                        (test[1]).reshape(-1,1),
                        (test[2]).reshape(-1,1)],1)
    return np.linalg.norm(f, ord='fro')/np.linalg.norm(div, ord='fro')

def NRMSE(pred, test, div):
    out = np.sqrt(np.mean(np.square(pred-test))/np.mean(np.square(div)))
    return out

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

def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def acc_cal(dynamic_params, all_params, g_batch, model_fns):
    all_params["network"]["layers"] = dynamic_params
    weights = all_params["problem"]["loss_weights"]
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]

    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["data"]["domain_range"]["t"][1]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["data"]["domain_range"]["t"][1]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["data"]["domain_range"]["t"][1]

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["data"]["domain_range"]["x"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["data"]["domain_range"]["x"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["data"]["domain_range"]["x"][1]
    px = all_params["data"]['u_ref']*out_x[:,3:4]/all_params["data"]["domain_range"]["x"][1]

    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["data"]["domain_range"]["y"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["data"]["domain_range"]["y"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["data"]["domain_range"]["y"][1]
    py = all_params["data"]['u_ref']*out_y[:,3:4]/all_params["data"]["domain_range"]["y"][1]

    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["data"]["domain_range"]["z"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["data"]["domain_range"]["z"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["data"]["domain_range"]["z"][1]
    pz = all_params["data"]['u_ref']*out_z[:,3:4]/all_params["data"]["domain_range"]["z"][1]
    
    acc_x = ut + u*ux + v*uy + w*uz
    acc_y = vt + u*vx + v*vy + w*vz
    acc_z = wt + u*wx + v*wy + w*wz
    acc = np.concatenate([acc_x.reshape(-1,1), acc_y.reshape(-1,1), acc_z.reshape(-1,1)],1)
    return acc
#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob
    checkpoint_fol = "DUCT_run1"
    #parser = argparse.ArgumentParser(description='Rwall_PINN')
    #parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    #args = parser.parse_args()
    #checkpoint_fol = args.checkpoint
    #print(checkpoint_fol, type(checkpoint_fol))
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a["data_init_kwargs"]['path'] = "/scratch/hyun/DUCT/TR_ppp_0_200_train/"
    a["problem_init_kwargs"]['path_s'] = "/scratch/hyun/DUCT/TR_ppp_0_200_valid/"
    #with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
    #    pickle.dump(a,f)

    values = list(a.values())
    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    #checkpoint_list = np.sort(glob(run.c.model_out_dir+'/*.pkl'))
    #with open(run.c.model_out_dir + "saved_dic_720000.pkl","rb") as f:
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[3].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
    dynamic_params = all_params["network"]["layers"]
    indexes, counts = np.unique(valid_data['pos'][:,0], return_counts=True)
    indexes2, counts2 = np.unique(train_data['pos'][:,0], return_counts=True)
#%%
    print(counts, counts2)
#%%
    print(train_data['acc'].shape, valid_data['acc'].shape)
#%% temporal error는 51개의 시간단계에대해서 [:,0]는 velocity error, [:,1]은 pressure error
    temporal_error_vel_list = []
    temporal_error_acc_list = []
    temporal_error_acc_t_list = []
    acc_v_list = []
    acc_t_list = []
    c = 0
    c2 = 0
    for j in range(50):
        print(j)
        acc = np.concatenate([acc_cal(all_params["network"]["layers"], all_params, valid_data['pos'][c:c+counts[j]][10000*s:10000*(s+1)], model_fn) 
                              for s in range(valid_data['pos'][c:c+counts[j]].shape[0]//10000+1)],0)
        acc_t = np.concatenate([acc_cal(all_params["network"]["layers"], all_params, train_data['pos'][c2:c2+counts2[j]][10000*s:10000*(s+1)], model_fn) 
                              for s in range(train_data['pos'][c2:c2+counts2[j]].shape[0]//10000+1)],0)
        pred = np.concatenate([model_fn(all_params, valid_data['pos'][c:c+counts[j]][10000*s:10000*(s+1)]) 
                              for s in range(valid_data['pos'][c:c+counts[j]].shape[0]//10000+1)],0)
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}

        output_ext = {output_keys[i]:valid_data['vel'][c:c+counts[j],i] for i in range(len(output_keys)-1)}
        output_ext_acc = {output_keys[i]:valid_data['acc'][c:c+counts[j],i] for i in range(len(output_keys)-1)}
        output_ext_acc_t = {output_keys[i]:train_data['acc'][c2:c2+counts2[j],i] for i in range(len(output_keys)-1)}
        c = c + counts[j]
        c2 = c2 + counts2[j]
        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                              output_ext['w'].reshape(-1,1)],1)
        f2 = np.concatenate([(acc[:,0]-output_ext_acc['u']).reshape(-1,1), 
                             (acc[:,1]-output_ext_acc['v']).reshape(-1,1), 
                             (acc[:,2]-output_ext_acc['w']).reshape(-1,1)],1)
        div2 = np.concatenate([output_ext_acc['u'].reshape(-1,1), output_ext_acc['v'].reshape(-1,1), 
                               output_ext_acc['w'].reshape(-1,1)],1)
        f3 = np.concatenate([(acc_t[:,0]-output_ext_acc_t['u']).reshape(-1,1), 
                             (acc_t[:,1]-output_ext_acc_t['v']).reshape(-1,1), 
                             (acc_t[:,2]-output_ext_acc_t['w']).reshape(-1,1)],1)
        div3 = np.concatenate([output_ext_acc_t['u'].reshape(-1,1), output_ext_acc_t['v'].reshape(-1,1), 
                               output_ext_acc_t['w'].reshape(-1,1)],1)

        temporal_error_vel_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))    
        temporal_error_acc_list.append(np.linalg.norm(f2, ord='fro')/np.linalg.norm(div2,ord='fro'))
        temporal_error_acc_t_list.append(np.linalg.norm(f3, ord='fro')/np.linalg.norm(div3,ord='fro'))
        acc_t_list.append(acc_t)
        acc_v_list.append(acc)
        print(acc_t.shape, acc.shape)
    temporal_error = np.concatenate([np.array(temporal_error_vel_list).reshape(-1,1),
                                     np.array(temporal_error_acc_list).reshape(-1,1),
                                     np.array(temporal_error_acc_t_list).reshape(-1,1)],1)
    acc_t_list = np.concatenate(acc_t_list)
    acc_t_list = np.concatenate([acc_t_list,train_data['acc']],1)
    acc_v_list = np.concatenate(acc_v_list)
    acc_v_list = np.concatenate([acc_v_list,valid_data['acc']],1)
#%%
    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)

    with open("datas/"+checkpoint_fol+"/temporal_error.pkl","wb") as f:
        pickle.dump(temporal_error,f)
    f.close()
    with open("datas/"+checkpoint_fol+"/acc_t.pkl","wb") as f:
        pickle.dump(acc_t_list,f)
    f.close()
    with open("datas/"+checkpoint_fol+"/acc_v.pkl","wb") as f:
        pickle.dump(acc_v_list,f)
    f.close()   