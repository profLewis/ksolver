import numpy as np
from smoothn import *

class Ksolver(object):
  '''
  Kernel model solver (with regularisation)

  '''
  def __init__(self,d,s=None,isrobust=True,f0=None):
    # initialise
    d = d.copy()

    mask = d['mask'][0]
    d['Ross'] = ma.array(d['Ross'],mask=mask)
    d['Li'] = ma.array(d['Li'],mask=mask)

    # form kernel estimation matrix terms
    k11 = np.sum(np.sum(d['Ross'] * d['Ross'],axis=1),axis=0)
    k22 = np.sum(np.sum(d['Li'] * d['Li'],axis=1),axis=0)
    k12 = np.sum(np.sum(d['Ross'] * d['Li'],axis=1),axis=0)
    k1 =  np.sum(np.sum(d['Ross'],axis=1),axis=0)
    k2 =  np.sum(np.sum(d['Li'],axis=1),axis=0)

    det = k11*k22 - k12*k12
    #det = N * det1 - k1*k1*k22 - k2*k2*k11 + 2 * k1*k2*k12

    M = np.zeros((2,2)).astype(object)
 
    M[0][0] =           k22
    M[0][1] = M[1][0] = -k12
    M[1][1]           = k11  

    # initial estimate at f0
    d['mask'] = (d['refl']>1.) | (d['refl']<0.) | d['mask']
    d['refl'] = ma.array(d['refl'],mask= d['mask'])
    if f0 == None:
      f0 = d['refl'].mean(axis=1)
      sz = smoothn(f0,axis=1,s=s,isrobust=isrobust)
      self.s = sz[1]
      self.f0 = sz[0]
      self.sz = sz
      self.outlier_wt = self.sz[3]
    else:
      self.f0 = f0

    self.nsets = d['refl'].shape[1]
    self.nbands = d['refl'].shape[0]

    self.M = M
    self.det = det
    self.mask = mask
    self.d = d
    self.isrobust = isrobust
    self.model = np.ones_like(d['refl'])

  def iterate(self):
    d = self.d
    f0 = self.f0
    nbands = self.nbands
    nsets = self.nsets
    M = self.M
    det = self.det
    s = self.s

    # fk is the angular part
    # rho/f0 - 1 = f1 k1 + f2 k2
    fk = ma.array(np.zeros_like(d['refl']),mask=d['mask'])
    for i in xrange(nsets):
        fk[:,i] = d['refl'][:,i]/f0 - 1.0

    f1 = ma.array(np.zeros_like(f0[:,0,:,:]))
    f2 = ma.array(np.zeros_like(f0[:,0,:,:]))

    P = [f1,f2]

    # solve for estimate of f1, f2
    # given f0
    # for each band
    for i in xrange(nbands):
      V = np.zeros(2).astype(object)
      # for each sample
      for j in xrange(nsets):
        V[0] += np.sum(fk[i,j] * d['Ross'][j],axis=0)
        V[1] += np.sum(fk[i,j] * d['Li'][j],axis=0)

      for k in xrange(2):
        P[k][i] = (V[0]*M[k][0] + V[1]*M[k][1])/det

    # now model 1 + f1 k1 + f2 k2
    model = np.zeros_like(d['refl'])
    for i in xrange(nsets):
      # bands
      for j in xrange(nbands):
        model[j,i] = 1.0 + P[0][j]*d['Ross'][i] + P[1][j]*d['Li'][i]

    model = ma.array(model,mask=d['mask'])

    # re-estimate f0 = rho/(1+f1 k1 + f2 k2)
    # and average over all samples for given band / location / day
    f0_old = f0
    f0 = (d['refl']/model).mean(axis=1)
    # and smooth it
    sz = smoothn(f0,z0=f0_old,axis=1,s=s,isrobust=self.isrobust)
    s = sz[1]
    f0 = sz[0]
    self.outlier_wt = self.sz[3]
    self.f0 = f0
    self.sz = sz
    self.s = s
    self.model = model
    self.P = P

  def plot_t(self,r=50,c=50,b=1):
    d = self.d
    f0 = self.f0
    cs = ['r','g','b','k']
    plt.clf()
    plt.plot(f0[b,:,r,c])
    plt.title('row %d col %d band %d'%(r,c,b))
    for i in xrange(d['refl'].shape[1]):
      plt.plot(d['refl'][b,i,:,r,c],'%s+'%cs[i])
      try:
        model = self.model[b,i,:,r,c]
      except:
        model = 1.0

      plt.plot(f0[b,:,r,c]*(model),'%sx'%cs[i])
    plt.show()


  def plot_s(self,r=50,c=50,b=1):
    from numpy import linalg
    d = self.d
    f0 = self.f0
    cs = ['r','g','b','c','m','y','k']
    sym = ['+','x','v','<','>','*','^']
    plt.clf()
    X = np.array([])
    Y = np.array([])
    W = np.array([])
    for b in xrange(d['refl'].shape[0]):
      for i in xrange(d['refl'].shape[1]):
        try:
          model = self.model[b,i,:,r,c]
        except:
          model = 1.0
        m = ~d['refl'][b,i,:,r,c].mask
        x = np.array(d['refl'][b,i,:,r,c][m])
        y = np.array((f0[b,:,r,c]*(model))[m])
        try:
          w = self.outlier_wt[b,:,r,c][m]
        except:
          w = np.ones_like(x)
  
        X = np.concatenate((X,x),axis=0)
        Y = np.concatenate((Y,y),axis=0)
        W = np.concatenate((W,w),axis=0)
        plt.plot(x,y,'%s%s'%(cs[b],sym[i]))
    d = (X-Y)*W
    rmse = np.sqrt(np.mean(d**2)/np.mean(W**2))
    b = np.sum(X**2 * W)
    a = np.sum(W)
    c = np.sum(X*W)
    d = np.sum(Y*W)
    e = np.sum(X*Y*W)
    det = a*b-c*c
    p = np.array([b*d - c*e, a*e - c*d])/det
    #p = linalg.lstsq(np.array([np.ones_like(X),X]).T,Y)[0]
    xx = np.array((X.min(),X.max()))
    yy = p[0] + xx * p[1]
    plt.plot(xx,yy,'k')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot((0.,1.),(0.,1.),'k--')
    plt.title('row %d col %d RMSE %.3f offset %.2f slope %.2f'%(r,c,rmse,p[0],p[1]))
    plt.show()

from ksolver import *

def ksolve(d,drmse_thresh = 0.02, doPlot=False,\
 		maxit=10,verbose=False):
 
  '''
  Run this to get fast(ish) BRDF modelling
  with kernel BRDF models and regularisation 
  with automatic estimation of smoothness.

  You will need numpy arrays of the following:

  d['mask'] : mask         shape: (nbands, nsets, ndays, nr, nc)
  d['refl'] : reflectance, shape: (nbands, nsets, ndays, nr, nc)
  d['Ross'] : Ross kernel, shape: (nsets, ndays, nr, nc)
  d['Li']   : Li kernel,   shape: (nsets, ndays, nr, nc)


  mask here is True for duff values and False elsewhere.

  nbands : number of wavebands
  nsets  : number of datasets
  ndays  : number of days
  nr     : number of rows
  nc     : number of columns

  '''
 
  k = Ksolver(d,isrobust=True)
  sse = 0.0
  sumW = 0.0
  for i in xrange(k.nsets):
    diff = (k.d['refl'][:,i] - k.f0)
    sse += np.sum(diff*diff*k.outlier_wt)
    sumW += np.sum(k.outlier_wt)
  rmse = [np.sqrt(sse/sumW)]
  if verbose:
    print -1,rmse
  if doPlot:
    plt.figure(0);k.plot_t(b=0);plt.figure(1);k.plot_s()
  for n in xrange(maxit):
    k.iterate()
    sse = 0.0
    sumW = 0.0
    for i in xrange(k.nsets):
      diff = (k.d['refl'][:,i] - k.f0*k.model[:,i])
      sse += np.sum(diff*diff*k.outlier_wt)
      sumW += np.sum(k.outlier_wt)
    rmse.append(np.sqrt(sse/sumW))
    drmse = (rmse[-2] - rmse[-1])/rmse[-2]
    if doPlot:
      plt.figure(0);k.plot_t(b=0);plt.figure(1);k.plot_s()
    k.d['rmse'] = rmse
    if verbose:
      print n,rmse,drmse
    if (n > 1) and (drmse < drmse_thresh):
      break
  return k

def __main__():
  import pickle

  pkl_file = open('data_nofire.pkl', 'rb')
  d = pickle.load(pkl_file)
  pkl_file.close()

  # just select 1 band
  #for k in ['mask','refl']:
  #  d[k] = d[k][1]
  #  d[k] = d[k].reshape((1,) + d[k].shape)

  k = ksolve(d)



