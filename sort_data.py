import numpy as np
from smoothn import *
from kernels import *
import re
data = {}
info = """
x brdf_2004_b01_fire.npz
x brdf_2004_b01_no_fire.npz
x brdf_2004_b02_fire.npz
x brdf_2004_b02_no_fire.npz
x brdf_2004_b03_fire.npz
x brdf_2004_b03_no_fire.npz
x brdf_2004_b04_fire.npz
x brdf_2004_b04_no_fire.npz
x brdf_2004_b05_fire.npz
x brdf_2004_b05_no_fire.npz
x brdf_2004_b06_fire.npz
x brdf_2004_b06_no_fire.npz
x brdf_2004_b07_fire.npz
x brdf_2004_b07_no_fire.npz
x SensorAzimuth_2004_fire.npz
x SensorAzimuth_2004_no_fire.npz
x SensorZenith_2004_fire.npz
x SensorZenith_2004_no_fire.npz
x SolarAzimuth_2004_fire.npz
x SolarAzimuth_2004_no_fire.npz
x SolarZenith_2004_fire.npz
x SolarZenith_2004_no_fire.npz
x statekm_2004_fire.npz
x statekm_2004_no_fire.npz
x the_doys.npz
""".split()[1::2]

info = [i.split('.')[0] for i in info]

for i in info:
    k,data[i] = np.load(i+'.npz').items()[0]
refl_fire = []
refl_nofire = []
c = re.compile('brdf')
for b in np.sort(data.keys()):
    if c.match(b):
        band = b.split('_')[2]
        if b.split('_')[-2] == 'no':
            refl_nofire.append(ma.array(data[b],mask=~data['statekm_2004_no_fire']))
        else:
            refl_fire.append(ma.array(data[b],mask=~data['statekm_2004_fire']))        

refl_nofire = ma.array(refl_nofire)
refl_fire = ma.array(refl_fire)

SensorAzimuth_2004_fire = ma.array(data['SensorAzimuth_2004_fire'],mask=~data['statekm_2004_fire'])
SensorAzimuth_2004_no_fire = ma.array(data['SensorAzimuth_2004_no_fire'],mask=~data['statekm_2004_no_fire'])
SensorZenith_2004_fire = ma.array(data['SensorZenith_2004_fire'],mask=~data['statekm_2004_fire'])
SensorZenith_2004_no_fire = ma.array(data['SensorZenith_2004_no_fire'],mask=~data['statekm_2004_no_fire'])
SolarAzimuth_2004_fire = ma.array(data['SolarAzimuth_2004_fire'],mask=~data['statekm_2004_fire'])
SolarAzimuth_2004_no_fire = ma.array(data['SolarAzimuth_2004_no_fire'],mask=~data['statekm_2004_no_fire'])
SolarZenith_2004_fire = ma.array(data['SolarZenith_2004_fire'],mask=~data['statekm_2004_fire'])
SolarZenith_2004_no_fire = ma.array(data['SolarZenith_2004_no_fire'],mask=~data['statekm_2004_no_fire'])

doys = data['the_doys']



mindoy = None
maxdoy = None

# unqiue doys
udoys = np.sort(np.unique(doys))
mindoy = mindoy or doys.min()
maxdoy = maxdoy or doys.max()
i = 0
m = np.array([(doys == udoys[i]).sum() for i in xrange(len(udoys))])
max_views = np.max(m)

s = refl_nofire.shape
all_doys = np.arange(mindoy,maxdoy+1)
refl_nofire_ = ma.array(np.zeros((s[0],max_views,len(all_doys),s[-2],s[-1])))
refl_fire_ = ma.array(np.zeros_like(refl_nofire_))
SensorAzimuth_2004_fire_ = ma.array(np.zeros((max_views,len(all_doys),s[-2],s[-1])))
SensorAzimuth_2004_no_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SensorZenith_2004_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SensorZenith_2004_no_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SolarAzimuth_2004_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SolarAzimuth_2004_no_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SolarZenith_2004_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))
SolarZenith_2004_no_fire_ = ma.array(np.zeros_like(SensorAzimuth_2004_fire_))

j = 0
for j in xrange(len(all_doys)):
    w = np.where(doys == all_doys[j])[0]
    for k,n in enumerate(w):
        refl_nofire_[:,k,j,:,:] = refl_nofire[:,n,:,:]
        refl_fire_[:,k,j,:,:] = refl_fire[:,n,:,:]
        SensorAzimuth_2004_fire_[k,j,:,:] = SensorAzimuth_2004_fire[n,:,:]
        SensorAzimuth_2004_no_fire_[k,j,:,:] = SensorAzimuth_2004_no_fire[n,:,:]
        SensorZenith_2004_fire_[k,j,:,:] = SensorZenith_2004_fire[n,:,:]
        SensorZenith_2004_no_fire_[k,j,:,:] = SensorZenith_2004_no_fire[n,:,:]
        SolarAzimuth_2004_fire_[k,j,:,:] = SolarAzimuth_2004_fire[n,:,:]
        SolarAzimuth_2004_no_fire_[k,j,:,:] = SolarAzimuth_2004_no_fire[n,:,:]
        SolarZenith_2004_fire_[k,j,:,:] = SolarZenith_2004_fire[n,:,:]
        SolarZenith_2004_no_fire_[k,j,:,:] = SolarZenith_2004_no_fire[n,:,:]
    
refl_nofire_ = ma.array(refl_nofire_/10000.)
refl_fire_ = refl_fire_/10000.
SensorAzimuth_2004_fire_ /= 100.
SensorAzimuth_2004_no_fire_ /= 100.
SensorZenith_2004_fire_ /= 100.
SensorZenith_2004_no_fire_ /= 100.
SolarAzimuth_2004_fire_ /= 100.
SolarAzimuth_2004_no_fire_ /= 100.
SolarZenith_2004_fire_ /= 100.
SolarZenith_2004_no_fire_ /= 100.

# calculate kernels
kkk = Kernels(SensorZenith_2004_fire_,SolarZenith_2004_fire_,\
              SensorAzimuth_2004_fire_-SolarAzimuth_2004_fire_,\
              RossHS=False,MODISSPARSE=True,RecipFlag=True,\
             normalise=1,doIntegrals=False,LiType='Sparse',RossType='Thick')
Ross_fire = kkk.Ross
Li_fire   = kkk.Li
# calculate kernels
kkk = Kernels(SensorZenith_2004_no_fire_,SolarZenith_2004_no_fire_,\
              SensorAzimuth_2004_no_fire_-SolarAzimuth_2004_no_fire_,\
              RossHS=False,MODISSPARSE=True,RecipFlag=True,\
             normalise=1,doIntegrals=False,LiType='Sparse',RossType='Thick')
Ross_no_fire = kkk.Ross
Li_no_fire   = kkk.Li


import pickle
data1 = {'Ross':np.array(Ross_no_fire),'Li':np.array(Li_no_fire),\
         'SensorAzimuth':np.array(SensorAzimuth_2004_no_fire_),\
	 'SensorZenith':np.array(SensorZenith_2004_no_fire_),\
	 'SolarAzimuth':np.array(SolarAzimuth_2004_no_fire_),\
	 'refl':np.array(refl_nofire_),'mask':refl_nofire_.mask,'doys':all_doys}

pkl_file = open('data_nofire.pkl', 'wb')
pickle.dump(data1,pkl_file)
pkl_file.close()

data1 = {'Ross':np.array(Ross_fire),'Li':np.array(Li_fire),\
         'SensorAzimuth':np.array(SensorAzimuth_2004_fire_),\
         'SensorZenith':np.array(SensorZenith_2004_fire_),\
         'SolarAzimuth':np.array(SolarAzimuth_2004_fire_),\
         'refl':np.array(refl_fire_),'mask':refl_fire_.mask,'doys':all_doys}

pkl_file = open('data_fire.pkl', 'wb')
pickle.dump(data1,pkl_file)
pkl_file.close()



