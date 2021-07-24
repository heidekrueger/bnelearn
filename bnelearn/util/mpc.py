"""This module contains the implementation of our custom GPU-enabled solver
for batched constrained quadratic programs, based on Mehrotra's Predictor-Corrector-Method (MPC).

authors:
    Anne Christopher
    Stefan Heidekrüger (@heidekrueger)
    Paul Sutterer
"""

from typing import Tuple, Union

import torch
import numpy as np

# lots of pylint false positives for -tensor in this module
#pylint: disable = invalid-unary-operand-type

class MpcSolver():
    r"""A batched qp-solver that solves batches of QPs of the form

    min   0.5 x.T Q x + q.T x
    s.t.  Gx <= h
          Ax = b

    with problem sice n (i.e. :math:`x\in\mathbb{R}^n}`)  , `n_ineq` many inequality constraints
    and `n_eq` equality constraints.

    The device and dtype used by the solver will be inferred from Q.

    Args:
        Q (torch.Tensor of dimension (n_batches, n, n)): A batch of positive definite n-by-n matrices.
        q (Tensor of dimension (n_batches, n))
        G (Tensor of dimension (n_batches, n_ineq, n))
        h (Tensor of dimension (n_batches, n_ineq))
        A (Tensor of dimension (n_batches, n_eq, n))
        b (Tensor of dimension (n_batches, n_eq))
        refine: bool. When set to `True`, if after max_iter iterations there are 
            still batches with residuals over 1e-6, the algorithm will run for another max_iter iterations,
            up to two additional times.
        print_warning (bool): if True, will print warnings if after three runs of max_iter iterations,
            the algorithm still hasn't converged sufficiently.
    """
    def __init__(self, max_iter=20):
        self.max_iter = max_iter

        # problem parameters
        self.Q: torch.Tensor = None # objective quadratic term
        self.q: torch.Tensor = None # objective linear term
        self.G: torch.Tensor = None # inequality constraints LHS
        self.G_T: torch.Tensor = None # (transposed, saved as contiguous tensor)
        self.h: torch.Tensor = None # ineqaulity constraints RHS
        self.A: torch.Tensor = None # equality constraints LHS
        self.A_T: torch.Tensor = None # (transposed, saved as contiguous tensor)
        self.b: torch.Tensor = None # equality constraints RHS

        # internal variables
        self.x: torch.Tensor = None # primal decision variables
        self.s: torch.Tensor = None # primal slack variables associated with inequality constraints
        self.z: torch.Tensor = None # Lagrange multiplicators associated with inequality constraints
        self.y: torch.Tensor = None # Lagrange multiplicators of equality constraints

        self.J: torch.Tensor = None # Jacobian of the KKT system

        # problem dimensions
        self.n_batch: int = None
        self.n_x: int = None
        self.n_eq: int = 0
        self.n_ineq: int = 0
        self.n_dual_vars = None   # = n_eq + n_ineq = n_constraints
        self.n_primal_vars = None # n_x + n_ineq (i.e x and s)

        #TODO: tbd
        self.refine: bool = None


    def solve(self, Q: torch.Tensor, q: torch.Tensor,
                    G: torch.Tensor, h: torch.Tensor,
                    A: torch.Tensor  = None, b: torch.Tensor = None,
                    refine=False, print_warning=False):

        self.device = Q.device
        self.dtype = Q.dtype
        assert self.is_pd(Q), "Q is not p.d., but this is a requirement!"

        self._set_and_verify_parameters(Q, q, G, h, A, b)

        self.refine = refine
        self._set_initial_Jacobian()
        self._update_J_LU()

        self.x, self.s, self.z, self.y = self._solve_kkt(
            -self.q.unsqueeze(-1),
            torch.zeros(self.n_batch, self.n_ineq,
                        device=self.device, dtype=self.dtype).unsqueeze(-1),
            h.unsqueeze(-1),
            b.unsqueeze(-1) if self.n_eq > 0 else None)

        alpha_p = self.get_initial(-self.z)
        alpha_d = self.get_initial(self.z)
        self.s = -self.z+alpha_p*(torch.ones_like(self.z))
        self.z = self.z+alpha_d*(torch.ones_like(self.z))

        # main iterations
        self.x, self.s, self.z, self.y = self.mpc_opt(
            print_warning=print_warning)
        op_val = 0.5*torch.bmm(torch.transpose(self.x, dim0=2, dim1=1), torch.bmm(self.Q, self.x)) + \
            torch.bmm(torch.transpose(self.q.unsqueeze(-1), dim0=2, dim1=1), self.x)
        return self.x, op_val

    def _set_and_verify_parameters(self, Q: torch.Tensor, q: torch.Tensor,
                                         G: torch.Tensor, h: torch.Tensor,
                                         A: torch.Tensor, b: torch.Tensor):
        """Checks whether problem parameters are compatible, assigns them as
        instance fields,  and determines and sets dimensions."""
        # 2 dimensions, i.e. no batches ==> dimensions are (n_ineq, n_x), add dimension n_batch at pos 0
        if(Q.dim() == G.dim() == 2):
            Q = Q.unsqueeze(0)
            q = q.unsqueeze(0)
            G = G.unsqueeze(0)
            h = h.unsqueeze(0)
        if (A is not None) and (A.dim() == 2):
            A = A.unsqueeze(0)
            b = b.unsqueeze(0)

        self.Q, self.q = Q, q
        self.G, self.h = G, h
        self.G_T = torch.transpose(G, dim0=2, dim1=1).contiguous()
        self.A, self.b = A, b
        if A is not None:
            self.A_T = torch.transpose(A, dim0=2, dim1=1).contiguous()

        # get sizes
        self.n_batch, self.n_ineq, self.n_x = self.G.size()
        if self.A is not None:
            n_batch_A, self.n_eq, n_x_A = self.A.size()
            assert n_batch_A == self.n_batch and n_x_A == self.n_x, \
                "batch and decision variables don't match between A and G!"
        else:
            self.n_eq = 0

        self.n_dual_vars = self.n_ineq + self.n_eq
        self.n_primal_vars = self.n_x + self.n_ineq

    @staticmethod
    def is_pd(Q):
        """checks whether Q (respectively, its entry-matrices in each batch,
        are positive definite.
        """
        try:
            torch.linalg.cholesky(Q)
            return True
        except RuntimeError as e:
            raise RuntimeError("Q is not PD") from e

    # TODO: following code is commented out while we confirm that the if
    # condition in the lower half can actually be removed.
    # def lu_factorize(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Returns the LU factorization of x.

    #     Args:
    #         x: torch.tensor (batch_size, n, n) (where n is arbitrary)

    #     """
    #     # We want to avoid pivoting if possible. As of pytorch 1.9,
    #     # this is possible only for cuda tensors.
    #     data, pivots = x.lu(pivot=not x.is_cuda)
    #     # define pivot matrix manually when on cuda
    #     #As of pytorch 1.9, this seems to be redundant,
    #     # as pivats returns this matrix by default. (let's keep the code around
    #     # to confirm nothing breaks.)
    #     if x.is_cuda == True:
    #         pivot matrix doesnt do any pivoting
    #         pivots = torch.arange(
    #             1, 1+x.size(1),
    #             dtype=torch.int, device=self.device
    #             ).unsqueeze(0).repeat(x.size(0), 1)
    #     return (data, pivots)

    def _set_initial_Jacobian(self):
        """Set up the initial Jacobian KKT matrix as the following block matrix:

            | Q 0 | Gt At |
            | 0 D |  I  0 |    | B1 | B2 |
        K = |--------------- = |----------
            | G I |  0  0 |    | B3 | B4 |
            | A 0 |  0  0 |
        Note that B2=transpose(B3).

        All entries except D will remain constant in all iterations.
        At initialization, we have D = I.
        """
        B1 = torch.zeros((self.n_batch, self.n_primal_vars, self.n_primal_vars),
                         device=self.device, dtype=self.dtype)
        B1[:, :self.n_x, :self.n_x] = self.Q
        # D here is unit identity matrix (initial case)
        B1[:, -self.n_ineq:, -self.n_ineq:] = torch.eye(
            self.n_ineq, device=self.device, dtype=self.dtype).repeat(self.n_batch, 1, 1)

        B3 = torch.zeros((self.n_batch, self.n_dual_vars, self.n_primal_vars),
                         device=self.device, dtype=self.dtype)
        B3[:, :self.n_ineq, :self.n_x] = self.G
        if self.n_eq > 0:
            B3[:, -self.n_eq:, :self.n_x] = self.A
        B3[:, :self.n_ineq, -self.n_ineq:] = torch.eye((self.n_ineq),
            device=self.device, dtype=self.dtype).repeat(self.n_batch, 1, 1)

        B4 = torch.zeros(self.n_batch, self.n_dual_vars, self.n_dual_vars,
                         device=self.device, dtype=self.dtype)

        size = self.n_primal_vars + self.n_dual_vars # = self.n_x+ 2*self.n_ineq + self.n_eq
        self.J = torch.zeros(self.n_batch, size, size,
                        device=self.device, dtype=self.dtype)
        self.J[:, :self.n_primal_vars, :self.n_primal_vars] = B1
        # following line will force a copy and ensure contiguity of J
        self.J[:, :self.n_primal_vars, self.n_primal_vars:] = torch.transpose(B3, dim0=2, dim1=1)
        self.J[:, self.n_primal_vars:, :self.n_primal_vars] = B3
        self.J[:, self.n_primal_vars:, self.n_primal_vars:] = B4

    def _update_J(self, d=None):
        """Updates the KKT Jacobian by replacing the D block with diag(d).

        Args:
            d: torch.tensor (n_batch, n_ineq, 1)
        """
        if d is not None:
            assert d.shape == torch.Size([self.n_batch, self.n_ineq, 1]), \
                "d has unexpected shape."
            self.J[ :, self.n_x:self.n_primal_vars, self.n_x:self.n_primal_vars] = \
                torch.diag_embed(d.squeeze(-1))

    def _update_J_LU(self):
        """Update the LU decomposition of the Jacobian based on the current Jacobian"""
        # TODO Stefan: wrapper no longer needed (??)
        # replaced by direct torch.lu call below, but keep around just in case
        #self.J_lu, self.J_piv = self.lu_factorize(self.J)
        self.J_lu, self.J_piv = self.J.lu(pivot = not self.J.is_cuda)

    def _solve_kkt(self, rx, rs, rz, ry) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        """Solve the KKT system with jacobian self.J and RHS specified by rx,rs,rz,ry, i.e.
        J %*% delta = (rx,rs,rz,ry)

        Where delta decomposes into delta = (dx, ds, dz, dy)
        Note that ry might be None in case there are no equality constraints.

        Args:
            rx, rs, rz, ry (torch.Tensors): right hand side vectors for the primal and dual variables.
        """
        if ry is not None:
            rhs = torch.cat((rx, rs, rz, ry), dim=1)
        else:
            rhs = torch.cat((rx, rs, rz), dim=1)
        delta = rhs.lu_solve(self.J_lu, self.J_piv)
        dx = delta[:,         :self.n_x          , :]
        ds = delta[:, self.n_x:self.n_primal_vars, :]
        if self.n_eq > 0:
            dz = delta[:, self.n_primal_vars:-self.n_eq, :]
            dy = delta[:,         -self.n_eq:          , :]
        else:
            dz = delta[:, self.n_primal_vars:, :]
            dy = None
        return dx, ds, dz, dy

    # ALTERNATIVE SOLVE_KKT USING BLOCK ELIMINATION TECHNIQUE
    # this has been found to be slower in our settings.
    # For details see Anne Christopher's MSc Thesis (2020), section 5.3.2
    # def _solve_kkt(self,rx,rs,rz,ry):
    #   b1=torch.cat((rx,rs),dim=1)
    #   if ry!=None:
    #     b2=torch.cat((rz,ry),dim=1)
    #   else:
    #     b2=rz
    #   A11= self.J[:,:self.nx+self.nineq,:self.nx+self.nineq]
    #   A12= self.J[:,:self.nx+self.nineq,self.nx+self.nineq:]
    #   A21=  torch.transpose(A12,dim0=2, dim1=1)
    #   # self.J_lu,self.J_piv= self.lu_factorize(self.J)
    #   # U_A11= torch.linalg.cholesky(A11)
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

    def remove_nans(self, dx, ds, dz, dy):
        """When the tensor of a batch contains NaNs in its first row,
           replace that entire batch with zero entries.

           Note that this behavior differs from simply checking
           torch.where(x.isnan()).
        """
        wh = torch.where(dx[:, 0, :] != dx[:, 0, :])[0]  # find NaN positions
        dx[wh, :, :] = 0.0
        ds[wh, :, :] = 0.0
        dz[wh, :, :] = 0.0
        if self.n_eq > 0:
            dy[wh, :, :] = 0.0
        return dx, ds, dz, dy, wh

    def mpc_opt(self, print_warning=True):
        """Runs the main iterations of """
        #bat = np.array([i for i in range(self.n_batch)])
        n_iter = 0
        while n_iter <= 3:
            if n_iter > 0:
                print(n_iter)
                print("Refining solutions with second round of iterations")
            for i in range(self.max_iter):

                # Calculate Residuals and check for convergence
                rx = -(torch.bmm(self.G_T, self.z) +
                       torch.bmm(self.Q, self.x)+ self.q.unsqueeze(-1))
                if self.n_eq > 0:
                    rx -= torch.bmm(self.A_T, self.y)
                rs = -self.z
                rz = -(torch.bmm(self.G, self.x)+self.s-self.h.unsqueeze(-1))
                ry = -(torch.bmm(self.A, self.x)-self.b.unsqueeze(-1)) if self.n_eq > 0 else None


                residual_x = torch.abs(rx)
                # complementary slackness residual
                mu = torch.abs(torch.bmm(torch.transpose(self.s, dim0=2, dim1=1),
                                         self.z).sum(1))/self.n_ineq
                residual_z = torch.abs(rz)
                residual_y = torch.abs(ry) if self.n_eq > 0 else torch.zeros_like(residual_z)

                residuals = torch.cat([r.max().view(1) for r in [residual_x, mu, residual_z, residual_y]])

                try:
                    if (residuals < 1e-6).all():
                        # print("Early exit at iteration no:",i)
                        return(self.x, self.s, self.z, self.y)
                except Exception as e:
                    #print(bat[torch.isnan(residual_x.sum(1)).squeeze(1)])
                    raise RuntimeError("invalid residuals, NaNs introduced?") from e

                # 2. Affine step calculation
                # get modified Jacobian and its lu factorization
                d = self.z/self.s
                self._update_J(d)
                self._update_J_LU()
                dx_aff, ds_aff, dz_aff, dy_aff = self._solve_kkt(rx, rs, rz, ry)

                # affine step size calculation
                alpha = torch.min(self._calculate_step_size(self.z, dz_aff),
                                  self._calculate_step_size(self.s, ds_aff))

                # find sigma for centering in the direction of mu
                # This requires temporarily calculating the affine-only updates for s and z
                s_aff = self.s+alpha*ds_aff
                z_aff = self.z+alpha*dz_aff
                mu_aff = torch.abs(torch.bmm(torch.transpose(s_aff, dim0=2, dim1=1),
                                   z_aff).sum(1))/self.n_ineq                
                sigma = (mu_aff/mu)**3

                # find centering+correction steps
                rx.zero_()
                rs = ((sigma*mu).unsqueeze(-1).repeat(1,self.n_ineq, 1) - ds_aff*dz_aff)/self.s
                rz.zero_()
                if self.n_eq > 0: # already zero otherwise.
                    ry.zero_()
                dx_cor, ds_cor, dz_cor, dy_cor = self._solve_kkt(rx, rs, rz, ry)

                dx = dx_aff + dx_cor
                ds = ds_aff + ds_cor
                dz = dz_aff + dz_cor
                dy = dy_aff + dy_cor if self.n_eq > 0 else None

                # find update step size
                alpha = torch.min(
                    torch.ones(self.n_batch, device=self.device, dtype=self.dtype).view(self.n_batch, 1, 1),
                    0.99*torch.min(self._calculate_step_size(self.z, dz), self._calculate_step_size(self.s, ds)))

                dx, ds, dz, dy, wh = self.remove_nans(dx, ds, dz, dy)
                if len(wh) == self.n_batch: #all batches have NaNs in the update --> terminate
                    return(self.x, self.s, self.z, self.y)

                self.x += alpha*dx
                self.s += alpha*ds
                self.z += alpha*dz
                if self.n_eq > 0:
                    self.y += alpha*dy
                else:
                    self.y = None

                if i == self.max_iter-1 and (residuals > 1e-10).any() and print_warning:
                    print("mpc exit in iter", i)
                    print("no of mu not converged: ", len(mu[mu > 1e-10]))
                    print("mpc warning: Residuals not converged, need more itrations")
            if self.refine == False:
                return(self.x, self.s, self.z, self.y)
            else:
                n_iter += 1
        return(self.x, self.s, self.z, self.y)

    @staticmethod
    def _calculate_step_size(v, dv):
        """Find batch_wise step size in the direction of dv.
        The step size should be as small as possible while ensuring that
        -v/dv will be positive after the update step.
        
        If invalid values are encountered, the step size will be set to 1."""
        v = v.squeeze(2)
        dv = dv.squeeze(2)
        div = -v/dv
        ones = torch.ones_like(div)
        div = torch.where(torch.isinf(div), ones, div)
        div = torch.where(torch.isnan(div), ones, div)
        div[dv > 0] = max(1.0, div.max())
        return (div.min(1)[0]).view(v.size()[0], 1, 1)

    def get_initial(self, z):
        """get step size using line search for initialization"""
        n_batch, _, _ = z.size()
        dz = torch.ones_like(z)
        div = -z/dz
        alpha = torch.max(div, dim=1).values.view(n_batch, 1, 1)+1  # 0.00001
        return alpha.view(n_batch, 1, 1)
