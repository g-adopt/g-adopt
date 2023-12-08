/* Implements pyop2 least_squares_kernel, which
 * slope limits linearisations of the FEM function and all its derivatives
 * by applying bounds at the vertices. The bounds at the vertices are based
 * on the min and max value of the average of the function (or of one of its
 * derivatives being slope limited) in the adjacent cells. The linearisation
 * of u^(i,j) consists of a cell average of u^(i,j) (ith x-derivative, jth y-derivative)
 * and the further derivatives u^(i+1,j) and u^(i,j+1) which determine the x and y variation
 * of u^(i,j) throughout the cell. Note that for the cell average in this linearisation,
 * and for the cell averages u^(i,j) used in the vertex bounds, we always use the unlimited
 * (original) values - i.e. although u^(i,j) itself will be changed in the slope limiting of
 * u^(i-1,j) and of u^(i,j-1) this is not taking into account in the slope limiting of u^(i,j).
 * See eqn 48-50 of Kuzmin '14 http://dx.doi.org/10.1016/j.jcp.2013.04.032 (noting the
 * difference between \bar u (for limited) and u (original)).
 * The changes in the x-slope u^(i+1,j) does however also affect the linearisation of u^(i+1,j-1),
 * as u^(i+1,j) is the y-slope of u^(i+1, j-1) (and similarly the y-slope u^(i,j+1) is the x-slope of
 * u^(i-1,j+1)). The slope limiting thus becomes a coupled system of inequality constraints imposed
 * at the vertices which we turn into a constrained minimisation problem trying to minimize the L2-norm
 * change in the overall solution. This quadratic program we solve using the algorithm outlined in
 * the appendix of Hoteit '14 http://dx.doi.org/10.1002/nme.1172
 *
 * Further notes:
 * - only the inequality constraints on derivatives of the same order m: {u^(i,j) | i+j=m}
 *   are linearly dependent. If the Taylor-basis mass matrix is diagonal, we could in principle consider
 *   the optimisation separately and in sequence for different orders. In curved cells however, the mass matrix
 *   is not diagonal and the optimal choice for the derivatives of one order, minimizing
 *   its contribution to the L2-norm difference, might change if different order derivatives are changed
 * - to formulate the quadratic program we need to collect a linearly-indepent set of constraints, a subset
 *   of the total number of constraints which may be linearly dependent. Currently we do this through
 *   a QR factorisation of the complete contraint matrix T which could possibly be avoided by working out
 *   some logic about which constraints are linearly dependent on each other (e.g. in a quadrilateral the
 *   constraints in the opposite vertex are linearly dependent (if active at the same time)). This might get
 *   a bit complicated in 3D however.
 * - Currently only 2D quads are implemented. 3D hexes should require relative minor additions. Supporting
 *   (higher order) simplices (in the same code?) requires a bit more thought
 *
 * NOTE: this C code is written as a python %%-template where %%(degree), %%(nvertices) and %%(nfacets) will
 * be substituted. Any other use of %% (in modulo arithmetic) should use a double %% to escape it. */
#include <stdbool.h>
#include <stdlib.h>
#include <petscsys.h>
#include <assert.h>
// Using lapack linked to PETSc
#include <petscblaslapack.h>
// Fortran name mangling including LAPACK precision (e.g. dgeqp3 for *d*ouble)
// We also use LAPACKpotrf_ and LAPACKtrtri_ but these
// are used in, and therefore defined by PETSc itself already
#define LAPACKgeqp3_ PETSCBLAS(geqp3, GEQP3)
#define LAPACKggglm_ PETSCBLAS(ggglm, GGGLM)

// these are compile time constants (through python substitution) so that we can define arrays of this size
const int m = %(nvertices)d;
const int n = %(degree)d + 1;
const int nfacets = %(nfacets)d;
const PetscBLASInt n2=n*n;

// Given feasible point up, find largest alpha such that
//    unew=alpha*u + (1-alpha)*up is feasible
// For constraints ijk with is_active[ijk] it is assumed that
//  the contraint is satisfied by *both* u and up as an equality constraint
//  and thus by unew (without checking)
PetscScalar find_feasible(PetscScalar *u, PetscScalar *up, bool *is_active,
                          PetscScalar *umin, PetscScalar *umax, PetscScalar *u0,
                          bool *is_interior_facet) {
    PetscScalar alpha=1.;

    // 0<=i<m loops over vertices where n2-1 constraints are imposed each
    // ijk is the continuous index in the m x n x n matrix
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
            if ((k<n-1) && is_interior_facet[2+i%%2]) { // contribution to the right
              lhs += Tx*(u[jk+1]-up[jk+1]);
              rhs += Tx*up[jk+1];
            }
            if ((jk+n<n2) && is_interior_facet[i/2]) { // contribution from row below
              lhs += Ty*(u[jk+n]-up[jk+n]);
              rhs += Ty*up[jk+n];
            }
            // from the above we want: umin-rhs<=alpha*lhs<=umax-rhs
            if (lhs<umin[ijk]-rhs-1e-12) {
                // because umin-rhs<=0 -> lhs<0 -> umin-rhs/lhs >=0
                alpha = fmax(fmin((umin[ijk]-rhs)/lhs, alpha), 0.);
                // now alpha<=(umin-rhs)/lhs (and lhs<0) -> alpha*lhs >= umin-rhs which is what we want
            }
            if (lhs>umax[ijk]-rhs+1e-12) {
                // because umax-rhs>=0 -> lhs>0 -> umax-rhs/lhs >=0
                alpha = fmax(fmin((umax[ijk]-rhs)/lhs, alpha), 0.);
                // now alpha<=(umax-rhs)/lhs (and lhs>0) -> alpha*lhs <= umax-rhs which is what we want
            }
        }
    }
    return alpha;
}

// loop through constraints that are not active yet (skipping those already is_active)
// and check whether any of those are satisfied as equality constraint and mark those
// as active
void find_new_active(PetscScalar *u, bool *is_active,
                    PetscScalar *umin, PetscScalar *umax, PetscScalar *u0,
                    bool *is_interior_facet) {
    const double eps=1e-12;
    for (int i=0, ijk=0; i<m; i++, ijk++) {
        double Tx = -0.5 + (i %% 2); // (xi-xc)/dx
        double Ty = -0.5 + (i / 2); // (yi-yc)/dy
        for (int jk=0; jk<n2-1; jk++, ijk++) {
            if (is_active[ijk]) continue;
            int k = jk %% n;
            double lhs = 0.0;
            if ((k<n-1) && is_interior_facet[2+i %% 2]) { // contribution to the right
              lhs += Tx*u[jk+1];
            }
            if ((jk+n<n2) && is_interior_facet[i /2]) { // contribution from row below
              lhs += Ty*u[jk+n];
            }
            is_active[ijk] = fmin(lhs-(umin[ijk]-u0[jk]), (umax[ijk]-u0[jk])-lhs) < eps;
        }
    }
}

/* check active constraints and compute the Lagrange multipliers lambda_ijk s.t.
      gradu = \pm lambda_ijk T_ijk
  where gradu is the derivative of the L2 difference we are minimizing and
  \pm=- for lower bound, \pm=+ for upper bound active constraint s.t. \pm T_ijk
  is the normal pointing *outside* the feasible region.
  For negative lambda, the gradient points inwards and thus the constraint is optimal
  in the sense that we can only further minimize the L2 difference by going outside of the
  feasible region. For positive lambdas otoh, by dropping the constraint we can allow it to
  go inside the feasible region to find a more optimal solution. We choose to drop the contraint
  with the maximu lambda_ijk (only if it is positive). We return this maximum lambda so that
  we can check whether a sub-optimal constraint has been found. If the returned max_lambda is negative
  all constraints are already optimal, and we are done */
void find_least_optimal(PetscScalar *u, PetscScalar *gradu, bool *is_active,
                    PetscScalar *umin, PetscScalar *umax, PetscScalar *u0,
                    PetscScalar *max_lambda, bool *is_interior_facet) {
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
            if ((k<n-1) && is_interior_facet[2+i %% 2]) { // contribution to the right
              dot += Tx*gradu[jk+1]; norm += Tx*Tx;
              lhs += Tx*u[jk+1];
            }
            if ((jk+n<n2) && is_interior_facet[i /2])  { // contribution from row below
              dot += Ty*gradu[jk+n]; norm += Ty*Ty;
              lhs += Ty*u[jk+n];
            }
            double lambda = dot/sqrt(norm);
            if (lhs-(umin[ijk]-u0[jk])<eps) { // lower bound constraint is active
              lambda = -lambda;
            }
            if (lambda>*max_lambda) {
              *max_lambda = lambda; max_ijk = ijk;
            }
        }
    }
    if (*max_lambda>0) {
      is_active[max_ijk] = false;
    }
}

void least_squares_kernel(
    PetscScalar *u,  // the field to be limited, in taylor basis: n x n
    PetscScalar *u0, // a copy of this field (left unchanged): n x n
    PetscScalar *umin, // the min value u^(j,k)_i, in the corners i: m x n x n
    PetscScalar *umax, // the max value u^(j,k)_i, in the corners i: m x n x n
    PetscScalar *mass, // L2 mass matrix in Taylor basis: (n x n) x (n x n)
    bool *is_interior_facet, // whether each local facet is interior (1) or exterior (0): nfacets
    PetscBLASInt *iterations, // for debugging purposes: n/o of iterations (added onto input value)
    PetscBLASInt *info // if any LAPACK calls returns a neg. info value, the kernel immediately
                       // returns with it - returns 0 upon succesful completion of the kernel
) {
    // NOTE: matrices are stored column major: T[col][row]
    PetscScalar T[n2][m*n2], RUinv[n2][m*n2];
    PetscScalar Uinv[n2*n2];
    PetscScalar up[n2], b[m*n2], y[n2];

    *info = 0;
    char uplo='U', diag='N';
    LAPACKpotrf_(&uplo, &n2, mass, &n2, info);
    if (*info<0) return;
    memcpy(Uinv, mass, n2*n2*sizeof(PetscScalar));
    LAPACKtrtri_(&uplo, &diag, &n2, Uinv, &n2, info);
    if (*info<0) return;

    up[0] = u[0];
    memset(up+1, 0, (n2-1)*sizeof(PetscScalar));
    bool is_active[m*n2];
    memset(is_active, false, m*n2*sizeof(bool));
    double alpha = find_feasible(u, up, is_active, umin, umax, u0, is_interior_facet);
    if (alpha == 1.0) return;

    for (int jk=1; jk<n2; jk++) {
      up[jk] = alpha*u[jk];
    }
    find_new_active(up, is_active, umin, umax, u0, is_interior_facet);
    while (true) {
        (*iterations)++;
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
                if ((k<n-1) && is_interior_facet[2+i%%2]) T[jk+1][nactive] = Tx;
                if ((jk+n<n2) && is_interior_facet[i/2]) T[jk+n][nactive] = Ty;
            }
        }

        // QR decomposition of T
        PetscBLASInt jpvt[n2], nconstraints=nactive+1;
        PetscBLASInt lda=m*n2, lwork=4*n2;
        PetscScalar tau[n2], work[lwork];
        memset(jpvt, 0, n2*sizeof(PetscBLASInt));
        // see: https://dl.acm.org/doi/pdf/10.1145/290200.287637 for Rank-Revealing QR
        LAPACKgeqp3_(&nconstraints, &n2, T, &lda, jpvt, tau, work, &lwork, info);
        if (*info<0) return;
        // T_ij(k) = (QR)_ik where j(k) is column reordering in jpvt (NOTE: jpvt uses fortran indexing from 1)

        // minimize (u-u0)^T M (u-u0) s.t. T u = T up:
        //
        // M = U^T U (Cholesky)
        // y = U (u-u0), u = u0 + Uinv y
        // minimize y^Ty s.t. T Uinv y = T up - T u0 = T (up-u0)

        // (T U_inv)_ml = \sum_k T_mj(k) Uinv_j(k)l = \sum_k,i Q_ki R_ik Uinv_j(k)l
        // [T (up-u0)]_i = Q_ki R_ik [up-u0]_j(k)
        // so instead we now impose R_ik Uinv_j(k)l y_l = R_ik [up-u0]_j(k)

        // compute matrix-matrix product R Uinv including column reordering
        // NOTE that due to this reordering the result is *not* upper triangular
        int i;
        for (i=0; i < (nconstraints<n2 ? nconstraints : n2) && fabs(T[i][i]) > 1e-12; i++) {
          for (int k=0; k<n2; k++) RUinv[k][i] = 0; // zero entire row
          for (int k=i; k<n2; k++) {
            for (int l=jpvt[k]-1; l<n2; l++) {
              RUinv[l][i] += T[k][i] * Uinv[l*n2+jpvt[k]-1];
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
        LAPACKggglm_(&nconstraints, &rankA, &n2, Amat, &lda, RUinv, &lda, b, x, y, work, &lwork, info);
        if (*info<0) return;

        // compute u = u0 + Uinv y
        for (int i=0; i<n2; i++) {
          u[i] = u0[i];
          for (int j=i; j<n2; j++) {
            u[i] += Uinv[j*n2+i]*y[j];
          }
        }
        assert(fabs(u[0]-u0[0])<1e-12);

        alpha = find_feasible(u, up, is_active, umin, umax, u0, is_interior_facet);
        if (alpha<1.) {
            for (int jk=1; jk<n2; jk++) {
              up[jk] = alpha*u[jk] + (1-alpha)*up[jk];
            }
            find_new_active(up, is_active, umin, umax, u0, is_interior_facet);
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
          find_least_optimal(u, gradu, is_active, umin, umax, u0, &max_lambda, is_interior_facet);
          if (max_lambda<0.) break;
          memcpy(up, u, n2*sizeof(PetscScalar));
        }
    }
}
