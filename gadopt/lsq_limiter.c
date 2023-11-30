/* pseudo code:
 *
 * find_feasible(
   def find_feasible(u, up, alpha, is_active):
      for ijk in constraints:
         if is_active[ijk]: continue
         if u violates constraint ijk:
            compute alpha s.t. u = alpha*u + (1-alpha)*up satisfies contraint

   def find_new_active(u, is_active):
      for ijk in constraints:
         if not is_active[ijk] and u[ijk] is on the boundary:

   up = [0]*n2
   is_active[:] = false
   find_feasible(u, up, alpha, is_active)
   if alpha==1. return
   up = alpha*u + (1-alpha)*up
   find_new_active(up, is_active)
   while (alpha<1)
       assemble constraints based on is_active
       minimize (u-u0)^T M (u-u0) s.t. T u = T up
       find_feasible(u, up, alpha, is_active)
       if alpha<1: # not feasible
           up = alpha*u + (1-alpha)*up
           find_new_active(up, is_active)
       else:
           find_least_optimal(u, is_active, ijk)
           if no ijk found (all optimal or empty active set): return
           is_active[ijk] = false
       up = u

*/

#include <stdbool.h>
#include <petscsys.h>
#include <assert.h>
#include <petscblaslapack.h>
#define LAPACKgeqp3_ PETSCBLAS(geqp3, GEQP3)
#define LAPACKggglm_ PETSCBLAS(ggglm, GGGLM)

const int m = 4; // number of corners
const int n = %(degree)d + 1;
const PetscBLASInt n2=n*n;

// Given feasible point up, find largest alpha such that
//    unew=alpha*u + (1-alpha)*up is feasible
// For constraints ijk with is_active[ijk] it is assumed that
//  the contraint is satisfied by *both* u and up as an equality constraint
//  and thus by unew (without checking)
PetscScalar find_feasible(PetscScalar *u, PetscScalar *up, bool *is_active,
                          PetscScalar *umin, PetscScalar *umax, PetscScalar *u0) {
    PetscScalar alpha=1.;

    // 0<=i<m loops over vertices where n2-1 constraints are imposed each
    // ijk is continuos index in m x n x n matrix
    // note additional ijk++ because [:,n-1, n-1] is skipped in inner loop
    for (int i=0, ijk=0; i<m; i++, ijk++) {
        double Tx = -0.5 + (i %% 2); // (xi-xc)/dx
        double Ty = -0.5 + (i / 2); // (yi-yc)/dy
        for (int jk=0; jk<n2-1; jk++, ijk++) {
            if (is_active[ijk]) continue;
            int k = jk %% n;  // column index in n x n matrix
            // we want umin[ijk]-u0[jk] <= T (alpha u + (1-alpha) up)[ijk] <= umax[ijk]-u0[jk]
            // or: umin[ijk]-u0[jk]-T up[ijk] <= alpha T (u-up)[ijk] <= umax[ijk]-u0[jk]-T up[ijk]
            // we know umin[ijk] -u0[jk] <= T up <= umax[ijk]-u0[jk]
            //          umin[ijk] - u0[jk] - Tup <= 0 <= umax[ijk]-u0[jk] - Tup
            double lhs = 0.0;
            double rhs = u0[jk];
            if (k<n-1) { // contribution to the right
              lhs += Tx*(u[jk+1]-up[jk+1]);
              rhs += Tx*up[jk+1];
            }
            if (jk+n<n2) { // contribution from row below
              lhs += Ty*(u[jk+n]-up[jk+n]);
              rhs += Ty*up[jk+n];
            }
            // from the above we want: umin-rhs<=alpha*lhs<=umax-rhs
            if (lhs<umin[ijk]-rhs) {
                // because umin-rhs<=0 -> lhs<0 -> umin-rhs/lhs >=0
                alpha = fmax(fmin((umin[ijk]-rhs)/lhs, alpha), 0.);
                // now alpha<=(umin-rhs)/lhs (and lhs<0) -> alpha*lhs >= umin-rhs which is what we want
            }
            if (lhs>umax[ijk]-rhs) {
                // because umax-rhs>=0 -> lhs>0 -> umax-rhs/lhs >=0
                alpha = fmax(fmin((umax[ijk]-rhs)/lhs, alpha), 0.);
                // now alpha<=(umax-rhs)/lhs (and lhs>0) -> alpha*lhs <= umax-rhs which is what we want
            }
        }
    }
    return alpha;
}

void find_new_active(PetscScalar *u, bool *is_active,
                    PetscScalar *umin, PetscScalar *umax, PetscScalar *u0) {
    const double eps=1e-12;
    for (int i=0, ijk=0; i<m; i++, ijk++) {
        double Tx = -0.5 + (i %% 2); // (xi-xc)/dx
        double Ty = -0.5 + (i / 2); // (yi-yc)/dy
        for (int jk=0; jk<n2-1; jk++, ijk++) {
            if (is_active[ijk]) continue;
            int k = jk %% n;
            double lhs = 0.0;
            if (k<n-1) { // contribution to the right
              lhs += Tx*u[jk+1];
            }
            if (jk+n<n2) { // contribution from row below
              lhs += Ty*u[jk+n];
            }
            is_active[ijk] = fmin(lhs-(umin[ijk]-u0[jk]), (umax[ijk]-u0[jk])-lhs) < eps;
        }
    }
}

void find_least_optimal(PetscScalar *u, PetscScalar *gradu, bool *is_active,
                    PetscScalar *umin, PetscScalar *umax, PetscScalar *u0,
                    PetscScalar *max_lambda) {
    const double eps=1e-12;
    int max_ijk;
    *max_lambda = -1.;
    for (int i=0, ijk=0; i<m; i++, ijk++) {
        double Tx = -0.5 + (i %% 2); // (xi-xc)/dx
        double Ty = -0.5 + (i / 2); // (yi-yc)/dy
        for (int jk=0; jk<n2-1; jk++, ijk++) {
            if (!is_active[ijk]) continue;
            // lower and upper bound are the same: we should always keep this as eq. constraint
            if (umax[ijk]-umin[ijk] < eps) continue;
            int k = jk %% n;
            double dot = 0.0; // gradu dot T_ijk
            double norm = 0.0, lhs = 0.0;
            if (k<n-1) { // contribution to the right
              dot += Tx*gradu[jk+1]; norm += Tx*Tx;
              lhs += Tx*u[jk+1];
            }
            if (jk+n<n2) { // contribution from row below
              dot += Ty*gradu[jk+n]; norm += Ty*Ty;
              lhs += Ty*u[jk+n];
            }
            double lambda = dot/sqrt(norm);
            if (umax[ijk]-u0[jk]-lhs<eps) { // upper bound constraint is active
              lambda = -lambda;
            }
            if (lambda>*max_lambda) {
              *max_lambda = lambda; max_ijk = ijk;
            }
        }
    }
    if (max_lambda>0) {
      is_active[max_ijk] = false;
    }
}

void least_squares_kernel(
    PetscScalar *u,  // the field to be limited, in taylor basis: n x n
    PetscScalar *u0, // a copy of this field (left unchanged): n x n
    PetscScalar *umin, // the min value u^(j,k)_i, in the corners i: m x n x n
    PetscScalar *umax, // the max value u^(j,k)_i, in the corners i: m x n x n
    PetscScalar *mass // L2 mass matrix in Taylor basis: (n x n) x (n x n)
) {
    // NOTE: matrices are stored column major: T[col][row]
    PetscScalar T[n2][m*n2], RUinv[n2][m*n2];
    PetscScalar Uinv[n2*n2];
    PetscScalar up[n2], b[m*n2], y[n2];

    PetscBLASInt info;
    char uplo='U', diag='N';
    LAPACKpotrf_(&uplo, &n2, mass, &n2, &info);
    memcpy(Uinv, mass, n2*n2*sizeof(PetscScalar));
    LAPACKtrtri_(&uplo, &diag, &n2, Uinv, &n2, &info);

    up[0] = u[0];
    memset(up+1, 0, (n2-1)*sizeof(PetscScalar));
    bool is_active[m*n2];
    memset(is_active, false, m*n2*sizeof(bool));
    double alpha = find_feasible(u, up, is_active, umin, umax, u0);
    if (alpha == 1.0) return;

    for (int jk=1; jk<n2; jk++) {
      up[jk] = alpha*u[jk];
    }
    while (true) {
        find_new_active(up, is_active, umin, umax, u0);
        int nactive = 0;
        // assemble constraint matrix T
        // first constraint is simply imposing u^(0,0) to be unchanged
        T[0][0] = 1.;
        for (int jk=1; jk<n2; jk++) T[jk][0] = 0.;

        // other constraints are those in the active set
        for (int i=0, ijk=0; i<m; i++, ijk++) {
            double Tx = -0.5 + (i %% 2); // (xi-xc)/dx
            double Ty = -0.5 + (i / 2); // (yi-yc)/dy
            for (int jk=0; jk<n2-1; jk++, ijk++) {
                if (!is_active[ijk]) continue;
                int k = jk %% n;
                ++nactive;
                for (int j=0; j<n2; j++) T[j][nactive] = 0.;
                if (k<n-1) T[jk+1][nactive] = Tx;
                if (jk+n<n2) T[jk+n][nactive] = Ty;
            }
        }

        // QR decomposition of T
        PetscBLASInt jpvt[n2], nconstraints=nactive+1;
        PetscBLASInt lda=m*n2, lwork=4*n2;
        PetscScalar tau[n2], work[lwork];
        memset(jpvt, 0, n2*sizeof(PetscBLASInt));
        // see: https://dl.acm.org/doi/pdf/10.1145/290200.287637 for Rank-Revealing QR
        LAPACKgeqp3_(&nconstraints, &n2, T, &lda, jpvt, tau, work, &lwork, &info);
        // T_ij(k) = (QR)_ik where j(k) is column reordering in jpvt (NOTE: jpvt uses fortran indexing from 1)

        // minimize (u-u0)^T M (u-u0) s.t. T u = T up:
        //
        // M = U^T U (Cholesky)
        // y = U (u-u0), u = u0 + Uinv y
        // minimize y^Ty s.t. T Uinv y = T up - T u0 = T (up-u0)

        // (T U_inv)_ml = \sum_k T_mj(k) Uinv_j(k)l = \sum_k,i Q_ki R_ik Uinv_j(k)l
        // [T (up-u0)]_i = Q_ki R_ik [up-u0]_j(k)
        // so instead we now impose R_ik Uinv_j(k)l y_l = R_ik [up-u0]_j(k)

        // compute matrix-matrix product R Uinv - where we "undo" the column reordering of R by reordering rows of Uinv
        int i;
        for (i=0; i<nconstraints && fabs(T[i][i])>1e-16; i++) {
          for (int k=0; k<n2; k++) RUinv[k][i] = 0;
          for (int k=i; k<n2; k++) {
            for (int l=jpvt[k]-1; l<n2; l++) {
              RUinv[k][i] += T[k][i] * Uinv[l*n2+jpvt[k]-1];
            }
          }
        }
        nconstraints = i;

        // compute b_i = R_ik [up-u0]_j(k)
        for (int i=0; i<nconstraints; i++) {
          b[i] = 0.;
          for (int k=i; k<n2; k++) {
            int j = jpvt[k]-1;
            b[i] += T[k][i] * (up[j]-u0[j]);
          }
        }

        // minimize y^y s.t. RU y = b
        const PetscBLASInt rankA=0;
        const PetscScalar Amat[n2][0], x[0];
        LAPACKggglm_(&nconstraints, &rankA, &n2, Amat, &lda, RUinv, &lda, b, x, y, work, &lwork, &info);

        // compute u = u0 + Uinv y
        for (int i=0; i<n2; i++) {
          u[i] = u0[i];
          for (int j=i; j<n2; j++) {
            u[i] += Uinv[j*n2+i]*y[j];
          }
        }
        assert(u[0] == u0[0]);

        alpha = find_feasible(u, up, is_active, umin, umax, u0);
        if (alpha<1.) {
            for (int jk=1; jk<n2; jk++) {
              up[jk] = alpha*u[jk] + (1-alpha)*up[jk];
            }
        } else {
          // compute gradu = M (u-u0) = U^T y
          PetscScalar gradu[n2];
          for (int i=0; i<n2; i++) { // column index of U
            gradu[i] = 0.;
            for (int j=0; j<i+1; j++) { // row
              gradu[i] += mass[i*n2+j]*y[j];
            }
          }
          double max_lambda;
          find_least_optimal(u, gradu, is_active, umin, umax, u0, &max_lambda);
          if (max_lambda<0.) break;
          memcpy(up, u, n2*sizeof(PetscScalar));
        }
    }
}
