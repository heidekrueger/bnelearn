import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
import random
import warnings
# warnings.filterwarnings('error')
import time

class mpc_class():
  def __init__(self,max_iter=20):
    self.max_iter=max_iter

  def solve(self,Q,q,G,h,A,b,refine=False,print_warning=True):
    self.Q=Q
    self.q=q
    self.G=G
    self.h=h
    self.G_T=torch.transpose(self.G,dim0=2,dim1=1)
    self.A=A
    self.b=b
    if self.A != None:
      self.A_T=torch.transpose(self.A,dim0=2,dim1=1)
    self.nbatch, self.nx, self.nineq, self.neq = self.get_sizes()
    self.is_Q_pd()
    self.refine=refine 
    self.J=self.get_Jacobian()
    self.J=self.get_lu_J()
    #initial solution
    if self.neq!=None:
      self.b_unsqueezed=self.b.unsqueeze(-1)
    else:
      self.b_unsqueezed=None
    self.x,self.s,self.z,self.y=self.solve_kkt(-self.q.unsqueeze(-1),
                                               torch.zeros(self.nbatch,self.nineq).unsqueeze(-1).type_as(self.Q),
                                               self.h.unsqueeze(-1),self.b_unsqueezed)
    # alpha_p=self.get_step(-self.z,torch.ones(self.z.size()).type_as(self.z))
    # alpha_d=self.get_step(self.z,torch.ones(self.z.size()).type_as(self.z))
    alpha_p=self.get_initial(-self.z)
    alpha_d=self.get_initial(self.z)
    self.s=-self.z+alpha_p*(torch.ones(self.z.size()).type_as(self.z))
    self.z=self.z+alpha_d*(torch.ones(self.z.size()).type_as(self.z))
    #main iterations
    start = time.time()
    self.x,self.s,self.z,self.y=self.mpc_opt(print_warning=print_warning)
    op_val=0.5*torch.bmm(torch.transpose(self.x,dim0=2,dim1=1),
                         torch.bmm(self.Q,self.x))+torch.bmm(
                         torch.transpose(self.q.unsqueeze(-1),dim0=2,dim1=1),self.x)
    t = time.time() - start
    # print("Optimization - time taken:", t)
    return self.x, op_val

  def get_sizes(self):
    #2 dimensions ==> dimensions are (ninenq,nx), add dimension nbatch at pos 0
    if(self.Q.dim()==self.G.dim()==2):  
      self.Q=self.Q.unsqueeze(0)
      self.q=self.q.unsqueeze(0)
      self.G=self.G.unsqueeze(0)
      self.h=self.h.unsqueeze(0)
      if (self.A is not None) and (self.A.dim()==2):
        self.A=self.A.unsqueeze(0)
        self.b=self.b.unsqueeze(0)
    #get sizes
    nbatch, nineq, nx = self.G.size()
    if self.A is not None:
      _,neq,_=self.A.size()
    else:
      neq=None
    return nbatch,nx,nineq,neq
  
  def is_Q_pd(self):
    try:
        torch.cholesky(self.Q)
    except:
        raise RuntimeError("Q is not PD")
    pass
  def lu_factorize(self,x):
    #do lu factorization of x
    #avoid pivoting when possible, i.e when on cuda
    data, pivots = x.lu(pivot=not x.is_cuda)
    #define pivot matrix manually when on cuda 
    if x.is_cuda==True:
        #pivot matrix doesnt do any pivoting
        pivots = torch.arange(1, 1+x.size(1),).unsqueeze(0).repeat(x.size(0), 1).int().cuda()
    return (data, pivots)

  def get_diag_matrix(self,d):
    #return diagonal matrix with diagonal entries d
    nBatch, n, _ = d.size()
    Diag = torch.zeros(nBatch, n, n).type_as(d)
    I = torch.eye(n).repeat(nBatch, 1, 1).type_as(d).bool()
    Diag[I] = d.view(-1)
    return Diag
  
  def get_Jacobian(self):
    #get the jacobian kkt matrix as concatenation of 4 blocks, B2=transpose(B3)
    B1=torch.zeros(self.nbatch,self.nx+self.nineq,self.nx+self.nineq).type_as(self.Q)
    if self.neq==None:
        B3=torch.zeros(self.nbatch,self.nineq,self.nx+self.nineq).type_as(self.Q)
        B4=torch.zeros(self.nbatch,self.nineq,self.nineq).type_as(self.Q)
    else:
        B3=torch.zeros(self.nbatch,self.neq+self.nineq,self.nx+self.nineq).type_as(self.Q)
        B4=torch.zeros(self.nbatch,self.neq+self.nineq,self.neq+self.nineq).type_as(self.Q)

    B1[:,:self.nx,:self.nx]=self.Q
    #D here is unit identity matrix (initial case)
    self.D=torch.eye(self.nineq).repeat(self.nbatch,1,1).type_as(self.Q)
    B1[:,-self.nineq:,-self.nineq:]=self.D

    B3[:,:self.nineq,:self.nx]=self.G
    if self.A!=None:
      B3[:,-self.neq:,:self.nx]=self.A
    B3[:,:self.nineq,-self.nineq:]=torch.eye(self.nineq).repeat(self.nbatch,1,1).type_as(self.Q)

    B2=torch.transpose(B3, dim0=2, dim1=1)
    self.J=torch.cat((torch.cat((B1,B2),dim=2),torch.cat((B3,B4),dim=2)),dim=1)
    return self.J

  def get_lu_J(self,d=None):
    # the jacobian J is modified when d is specified
    if d!=None:
      self.D=self.get_diag_matrix(d)
      self.J[:,self.nx:self.nx+self.nineq,self.nx:self.nx+self.nineq]=self.D
    self.J_lu,self.J_piv= self.lu_factorize(self.J)
    return self.J

  def solve_kkt(self,rx,rs,rz,ry):
    # solve the KKT system with jacobian J and F specified by rx,rs,rz,ry
    if ry!=None:
      F=torch.cat((rx,rs,rz,ry), dim=1)
    else:
      F=torch.cat((rx,rs,rz), dim=1)
    step=F.lu_solve(self.J_lu,self.J_piv)
    dx=step[:,:self.nx,:]
    ds=step[:,self.nx:self.nx+self.nineq,:]
    if self.neq!=None:
      dz=step[:,self.nx+self.nineq:-self.neq,:]
      dy=step[:,-self.neq:,:]
    else:
      dz=step[:,self.nx+self.nineq:,:]
      dy=None
    return(dx,ds,dz,dy)
  
  # SOLVE_KKT USING BLOCK ELIMINATION TECHNIQUE 
  # def solve_kkt(self,rx,rs,rz,ry):
  #   b1=torch.cat((rx,rs),dim=1)
  #   if ry!=None:
  #     b2=torch.cat((rz,ry),dim=1)
  #   else:
  #     b2=rz
  #   A11= self.J[:,:self.nx+self.nineq,:self.nx+self.nineq]
  #   A12= self.J[:,:self.nx+self.nineq,self.nx+self.nineq:]
  #   A21=  torch.transpose(A12,dim0=2, dim1=1)
  #   # self.J_lu,self.J_piv= self.lu_factorize(self.J)
  #   # U_A11= torch.cholesky(A11)
  #   U_A11,U_A11_piv= A11.lu(pivot=False)
  #   # u=torch.cholesky_solve(b1,U_A11)
  #   u=torch.lu_solve(b1,U_A11,U_A11_piv)
  #   # v=torch.cholesky_solve(A12,U_A11)
  #   v=torch.lu_solve(A12,U_A11,U_A11_piv)
  #   S_neg=torch.bmm(A21,v)
  #   U_S_neg,U_S_neg_piv= S_neg.lu(pivot=False)
  #   # w= torch.cholesky_solve(b2,U_S_neg)
  #   w= torch.lu_solve(b2,U_S_neg,U_S_neg_piv)
  #   # t= torch.cholesky_solve(A21,U_S_neg )
  #   t= torch.lu_solve(A21,U_S_neg,U_S_neg_piv )
  #   x2= -(w-torch.bmm(t,u))
  #   x1= u - torch.bmm(v,x2)
  #   dx=x1[:,:self.nx,:]
  #   ds=x1[:,self.nx:,:]
  #   if ry!=None:
  #     dz=x2[:,:-self.neq,:]
  #   else:
  #     dz=x2
  #   if ry!=None:
  #     dy=x2[:,-self.neq:,:]
  #   else:
  #     dy=None
  #   return (dx,ds,dz,dy)

  def remove_nans(self,dx,ds,dz,dy):
    wh= torch.where(dx[:,0,:]!=dx[:,0,:])[0] #find NaN positions
    dx[wh,:,:]=torch.zeros(len(wh),dx.size()[1],dx.size()[2]).type_as(dx)
    ds[wh,:,:]=torch.zeros(len(wh),ds.size()[1],ds.size()[2]).type_as(dx)
    dz[wh,:,:]=torch.zeros(len(wh),dz.size()[1],dz.size()[2]).type_as(dx)
    if self.neq!=None:
      dy[wh,:,:]=torch.zeros(len(wh),dy.size()[1],dy.size()[2]).type_as(dx)
    return dx,ds,dz,dy,wh

  def mpc_opt(self,print_warning=True):
    count=0
    bat=np.array([i for i in range(self.nbatch)])
    n_iter=0
    this_problem_not_converged= torch.ones(self.nbatch).type_as(self.Q).view(self.nbatch,1,1)
    while (n_iter<=3):
      if n_iter>0:
        print(n_iter)
        print("Refining solutions with second round of iterations")
      for i in range(self.max_iter):
          if self.neq!=None:
            rx= -(torch.bmm(self.A_T,self.y)+torch.bmm(self.G_T,self.z)+torch.bmm(self.Q,self.x)+self.q.unsqueeze(-1))
          else:
            rx= -(torch.bmm(self.G_T,self.z)+torch.bmm(self.Q,self.x)+self.q.unsqueeze(-1))
          rs=-self.z
          rz=-(torch.bmm(self.G,self.x)+self.s-self.h.unsqueeze(-1))
          if self.neq!=None:
            ry=-(torch.bmm(self.A,self.x)-self.b.unsqueeze(-1))
          else:
            ry=None
          d=self.z/self.s
          mu=torch.abs(torch.bmm(torch.transpose(self.s,dim0=2,dim1=1),self.z).sum(1))/self.nineq
          pri_resid=torch.abs(rx)
          dual_1_resid=torch.abs(rz)
          if self.neq!=None:
            dual_2_resid=torch.abs(ry)
            resids=np.array([pri_resid.max(),mu.max(),dual_1_resid.max(),dual_2_resid.max()])
          else:
            dual_2_resid=torch.zeros(dual_1_resid.size()).type_as(dual_1_resid)
            resids=np.array([pri_resid.max(),mu.max(),dual_1_resid.max()])
          
          #find if any of the problems converged
          # if (mu < 1e-6).any() :
          #   where= torch.where(mu<1e-6)[0]
          #   p_resids=torch.max(pri_resid[where,:,:].view(len(where),-1),dim=1).values
          #   d1_resids=torch.max(pri_resid[where,:,:].view(len(where),-1),dim=1).values
          #   d2_resids=torch.max(pri_resid[where,:,:].view(len(where),-1),dim=1).values
          #   sum_resids=p_resids+d1_resids+d2_resids
          #   resids_where= where[torch.where(sum_resids<1e-6)[0] ]
          #   compareview = resids_where.repeat(where.shape[0],1).T
          #   this_problem_not_converged[where,:,:]=0  
          try:
            if (resids<1e-6).all():
              # print("Early exit at iteration no:",i)
              return(self.x,self.s,self.z,self.y)
          except:
            print(bat[torch.isnan(pri_resid.sum(1)).squeeze(1)])
            raise RuntimeError("invalid res")
          
          #affine step calculation
          #get modified Jacobian and its lu factorization
          self.J=self.get_lu_J(d)
          dx_aff,ds_aff,dz_aff,dy_aff=self.solve_kkt(rx,rs,rz,ry)

          #affine step size calculation
          alpha = torch.min(self.get_step(self.z, dz_aff),self.get_step(self.s, ds_aff))
          
          #affine updates for s and z
          s_aff=self.s+alpha*ds_aff
          z_aff=self.z+alpha*dz_aff
          mu_aff=torch.abs(torch.bmm(torch.transpose(s_aff,dim0=2,dim1=1),z_aff).sum(1))/self.nineq
          
          #find sigma for centering in the direction of mu
          sigma=(mu_aff/mu)**3

          #find centering+correction steps
          rx=torch.zeros(rx.size()).type_as(self.Q)
          rs=((sigma*mu).unsqueeze(-1).repeat(1,self.nineq,1)-ds_aff*dz_aff)/self.s
          rz=torch.zeros(rz.size()).type_as(self.Q)
          if self.neq!=None:
            ry=torch.zeros(ry.size()).type_as(self.Q)
          dx_cor,ds_cor,dz_cor,dy_cor=self.solve_kkt(rx,rs,rz,ry)

          dx=dx_aff+dx_cor
          ds=ds_aff+ds_cor
          dz=dz_aff+dz_cor
          if self.neq!=None:
            dy=dy_aff+dy_cor
          else:
            dy=None
          # find update step size
          alpha = torch.min(torch.ones(self.nbatch).type_as(self.Q).view(self.nbatch,1,1),0.99*torch.min(self.get_step(self.z, dz),self.get_step(self.s, ds)))
          
          #check for early exit
          # if torch.isnan(dx).all():
          #   return(self.x,self.s,self.z,self.y)

          # update
          # dx[torch.isnan(dx)]=0
          # ds[torch.isnan(ds)]=0
          # dz[torch.isnan(dz)]=0
          # if self.neq!=None:
          #   dy[torch.isnan(dy)]=0
          dx,ds,dz,dy,wh= self.remove_nans(dx,ds,dz,dy)
          if len(wh)== self.nbatch:
            return(self.x,self.s,self.z,self.y)
          # dx[dx!=dx]=0
          # ds[ds!=ds]=0
          # dz[dz!=dz]=0
          # if self.neq!=None:
          #   dy[dy!=dy]=0
          # dx[torch.where(this_problem_not_converged==0)[0],:,:]=0
          # ds[torch.where(this_problem_not_converged==0)[0],:,:]=0
          # dz[torch.where(this_problem_not_converged==0)[0],:,:]=0
          # if self.neq!=None:
          #   dy[torch.where(this_problem_not_converged==0)[0],:,:]=0
          
          self.x+=alpha*dx
          self.s+=alpha*ds
          self.z+=alpha*dz
          if self.neq!=None:
            self.y+=alpha*dy
          else:
            self.y=None
          # do not update problems that already converged
          # self.x+=torch.bmm(alpha*dx, this_problem_not_converged)
          # self.s+=torch.bmm(alpha*ds, this_problem_not_converged)
          # self.z+=torch.bmm(alpha*dz, this_problem_not_converged)
          # if self.neq!=None:
          #   self.y+=torch.bmm(alpha*dy, this_problem_not_converged)
          # else:
          #   self.y=None
          # if (this_problem_not_converged==0).all():
            # print("All problems converged, exiting at iter ",i)
            # return(self.x,self.s,self.z,self.y)
          
          if(i==self.max_iter-1 and (resids>1e-10).any()) & print_warning==True:
            print("mpc exit in iter",i)
            print("no of mu not converged: ",len(mu[mu>1e-10]))
            # print("no of primal residual not converged: ",len(pri_resid[pri_resid>1e-10]))
            # print("no of dual residual 1 not converged: ",len(dual_1_resid[dual_1_resid>1e-10]))
            # print("no of dual residual 2 not converged: ",len(dual_2_resid[dual_2_resid>1e-10]))
            print("mpc warning: Residuals not converged, need more itrations")
      if self.refine==False:
        return(self.x,self.s,self.z,self.y)
      else:
        n_iter+=1
    return(self.x,self.s,self.z,self.y)

  def get_step(self,v,dv):
    v=v.squeeze(2)
    dv=dv.squeeze(2)
    div= -v/dv
    ones=torch.ones_like(div)
    div=torch.where(torch.isinf(div),ones,div)
    div=torch.where(torch.isnan(div),ones,div)
    div[dv>0]=max(1.0,div.max())
    return (div.min(1)[0]).view(v.size()[0],1,1)
  def get_initial(self,z):
      #get step size using line search for initialization 
      nbatch,_,_=z.size()
      dz=torch.ones(z.size()).type_as(z)
      div= -z/dz
      alpha=torch.max(div,dim=1).values.view(nbatch,1,1)+1#0.00001
      return alpha.view(nbatch,1,1)
      