#%%
import numpy as np
from glob import glob
class Database:
    @staticmethod
    def init_parmas(path, s_range, t_range, track_limit):
        raise NotImplementedError
    @staticmethod
    def data_load(filename):
        raise NotImplementedError
    @staticmethod
    def track_filter(track):
        raise NotImplementedError
    @staticmethod
    def domain_filter(s_trange):
        raise NotImplementedError
    @staticmethod
    def data_split(data, ratio):
        raise NotImplementedError
    
class Data(Database):
    @staticmethod
    #def init_params(path, domain_range, timeskip, track_limit, frequency, data_keys, viscosity):
    def init_params(**kwargs):
        data_params = {}
        for key, value in kwargs.items():
            data_params[key] = value
        return data_params
    @staticmethod
    def data_load(filename, data_keys):
        sscale, vscale, ascale = ([1, 1, 1], [1, 1, 1], [1, 1, 1])
        order = (0,1,2)
        sorder, vorder, aorder = (iter(order), iter(order), iter(order))
        with open(filename, 'r') as file:
            i=0
            while 1: 
                line = file.readline()
                i += 1
                if line.split(' ')[0] == 'Variables': break

            for var in line.split('"'):
                if var[0] == 'x' or var[0] == 'y' or var[0] == 'z':
                    if var[2:] == '[mm]': sscale[next(sorder)] = 0.001

                        
                    else: next(sorder)
                        
                if var[0] == 'u' or var[0] == 'v' or var[0] == 'w':
                    if var[2:] == '[mm/s]': vscale[next(vorder)] = 0.001
                    else: next(vorder)
                        
                if var[0:2] == 'ax' or var[0:2] == 'ay' or var[0:2] == 'az':
                    if var[3:] == '[mm/s^2]': ascale[next(aorder)] = 0.001
                    else: next(aorder)

            if line.split(' ')[0] == 'Zone':
                i+=1
        data = np.loadtxt(filename, skiprows=i)

        pos = np.concatenate([data[:,0:1]*sscale[0], data[:,1:2]*sscale[1], data[:,2:3]*sscale[2]],1)
        vel = np.concatenate([data[:,3:4]*vscale[0], data[:,4:5]*vscale[1], data[:,5:6]*vscale[2]],1)
        if 'acc' in data_keys:
            acc = np.concatenate([data[:,6:7]*ascale[0], data[:,7:8]*ascale[1], data[:,8:9]*ascale[2]],1)
        if 'track' in data_keys:
            track = data[:,9:10]
        all_data = {}
        for i in range(len(data_keys)):
            all_data[data_keys[i]] = eval(data_keys[i])
        return all_data

    @staticmethod
    def data_load_npy(filename, data_keys):
        data = np.load(filename)
        pos = data[:,0:3]*0.001
        vel = data[:,3:6]
        if 'acc' in data_keys:
            acc = data[:,6:9]
        if 'track' in data_keys:
            track = data[:,10:11]
        all_data = {}
        for i in range(len(data_keys)):
            all_data[data_keys[i]] = eval(data_keys[i])
        return all_data

    @staticmethod
    def track_filter(all_data_, data_keys, track_limit):
        index = np.where(all_data_['track']<=track_limit)
        all_data_ = {data_keys[i]:all_data_[data_keys[i]][index[0],:] for i in range(len(data_keys))}
        return all_data_

    @staticmethod
    def domain_filter(all_data_, data_keys, domain_range):
        index = np.where((all_data_['pos'][:,1]>=domain_range['x'][0])&(all_data_['pos'][:,1]<domain_range['x'][1])&
                         (all_data_['pos'][:,2]>=domain_range['y'][0])&(all_data_['pos'][:,2]<domain_range['y'][1])&
                         (all_data_['pos'][:,3]>=domain_range['z'][0])&(all_data_['pos'][:,3]<domain_range['z'][1]))
        all_data_ = {data_keys[i]:all_data_[data_keys[i]][index[0],:] for i in range(len(data_keys))}
        return all_data_
    
    @staticmethod
    def data_split(data, data_keys, ratio):
        train_data = {data_keys[i]:[] for i in range(len(data_keys))}
        valid_data = {data_keys[i]:[] for i in range(len(data_keys))}
        seed_number= np.arange(0,1000)
        np.random.seed(42)
        seeds = np.random.choice(seed_number, np.unique(data['pos'][:,0]).shape[0])
        for i, timestep in enumerate(np.unique(data['pos'][:,0])):
            np.random.seed(seeds[i])
            index_t = np.where(data['pos'][:,0]==timestep)
            index_r = np.random.choice(data['pos'][index_t[0],:].shape[0],data['pos'][index_t[0],:].shape[0])
            for j in range(len(data_keys)): train_data[data_keys[j]].append(data[data_keys[j]][index_t[0],:][index_r[:int(ratio*data[data_keys[j]][index_t[0],:].shape[0])],:])
            for k in range(len(data_keys)): valid_data[data_keys[k]].append(data[data_keys[k]][index_t[0],:][index_r[int(ratio*data[data_keys[k]][index_t[0],:].shape[0]):],:])

        for m in range(len(data_keys)): train_data[data_keys[m]] = np.concatenate(train_data[data_keys[m]], 0, dtype=np.float32)
        for n in range(len(data_keys)): valid_data[data_keys[n]] = np.concatenate(valid_data[data_keys[n]], 0, dtype=np.float32)
        return train_data, valid_data
    
    @staticmethod
    def input_normalize(all_params, data):
        domain_range = all_params["data"]["domain_range"]
        frequency = all_params["data"]["frequency"]
        arg_keys = ['t', 'x', 'y', 'z']
        for i in range(data['pos'].shape[1]):
            data['pos'][:,i] = data['pos'][:,i]/domain_range[arg_keys[i]][1]
        return data

    @staticmethod
    def output_normalize(all_params, data):
        vel_ref_keys = ['u_ref', 'v_ref', 'w_ref']
        vel_ref = {vel_ref_keys[i]:np.max(np.abs(data['vel'][:,i:i+1])) for i in range(len(vel_ref_keys))}
        all_params["data"].update(vel_ref)
        return all_params


    @staticmethod
    def train_data(all_params):
        path = all_params["data"]["path"]
        timeskip = all_params["data"]["timeskip"]
        domain_range = all_params["data"]["domain_range"]
        track_limit = all_params["data"]["track_limit"]
        frequency = all_params["data"]["frequency"]
        data_keys = all_params["data"]["data_keys"]
        vel_ref_keys = ['u_ref', 'v_ref', 'w_ref']
        if glob(path+"*.npy"):
            filenames = sorted(glob(path+'*.npy'))[int(domain_range['t'][0]*frequency):int(domain_range['t'][-1]*frequency)+1][::timeskip]
        else:
            filenames = sorted(glob(path+'*.dat'))[int(domain_range['t'][0]*frequency):int(domain_range['t'][-1]*frequency)+1][::timeskip]
        datas = {data_keys[i]:[] for i in range(len(data_keys))}
        for t, filename in enumerate(filenames):
            if ".dat" in filename:
                all_data_ = Data.data_load(filename, data_keys)
            if ".npy" in filename:
                all_data_ = Data.data_load_npy(filename, data_keys)
            if 'track' in data_keys:
                all_data_ = Data.track_filter(all_data_, data_keys, track_limit)
            time = np.zeros((all_data_['pos'].shape[0],1))+t*int(timeskip)/frequency
            all_data_['pos'] = np.concatenate([time,all_data_['pos']],1)
            for i in range(len(data_keys)): datas[data_keys[i]].append(all_data_[data_keys[i]])
        for j in range(len(data_keys)): datas[data_keys[j]] = np.concatenate(datas[data_keys[j]], 0, dtype=np.float32)
        #datas[data_keys[0]][:,1] = datas[data_keys[0]][:,1] + 0.6
        #datas[data_keys[0]][:,2] = datas[data_keys[0]][:,2] + 0.6
        datas = Data.domain_filter(datas, data_keys, domain_range)
        print(np.max(datas['pos'][:,1]))
        s = np.unique(datas['pos'][:,0])
        print(s[1]-s[0])
        if "data_split" in list(all_params["data"].keys()):
            train_data, valid_data = Data.data_split(datas, data_keys, 0.8)
            train_data = Data.input_normalize(all_params, train_data)
            valid_data = Data.input_normalize(all_params, valid_data)
        else:
            train_data = Data.input_normalize(all_params, datas)
        all_params = Data.output_normalize(all_params, train_data)
        #for i in range(train_data['vel'].shape[1]):
        #    train_data['vel'][:,i] = train_data['vel'][:,i]/all_params["data"][vel_ref_keys[i]]
        #    if "valid_data" in locals():
        #        valid_data['vel'][:,i] = valid_data['vel'][:,i]/all_params["data"][vel_ref_keys[i]]
        if "valid_data" in locals():
            return train_data, valid_data, all_params
        else:
            return train_data, all_params


if __name__ == "__main__":
    all_params = {"data":{}}
    #path = '/home/bussard/hyun_sh/TBL_PINN/data/HIT/Particles/'
    path = '/scratch/hyun/Comp/TR_ppp_0_120_FR_6_I0_train/'
    timeskip = 1
    track_limit = 424070
    frequency = 6250000
    domain_range = {'t':(0,51/frequency), 'x':(0,0.01225), 'y':(0,0.0023), 'z':(0,0.0012)}
    viscosity = 1*10**(-6)
    data_keys = ['pos', 'vel', 'acc']
    data_split = 0.8
    all_params["data"] = Data.init_params(path = path, domain_range = domain_range, 
                                          timeskip = timeskip, track_limit = track_limit, 
                                          frequency = frequency, data_keys = data_keys, 
                                          viscosity = viscosity)
    
    train_data, all_params = Data.train_data(all_params)
