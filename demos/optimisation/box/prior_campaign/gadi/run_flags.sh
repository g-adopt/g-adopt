# The flags of the Gauss-Newton box runs (NOTES/team/gauss-newton/PLAN.md section 4.2).
# submit_gn.sh writes them into the argument file of each run, and smoke_gn.pbs uses
# them with a smaller iteration limit, so the smoke runs the commands of the runs.
#
# All runs: 64 cells, window 80, the tutorial regularisation in the objective, first
# guess the observed temperature, budget 800 forward-equivalents, serial.
COMMON=(--window 80 --alpha-d 1e-3 --alpha-s 1e-3 --initial-guess observed
        --gradient-rtol 1e-6 --max-it 500 --work-budget 800)
# The Newton arms: TAO BNLS with the metric as Riesz map and preconditioner, Krylov
# cap 100, BNK's shift at zero, pinned walls and the Dirichlet operator.
NEWTON=(--optimiser tao-bnls-riesz --riesz regularisation --krylov-max-it 100
        --tao-opt tao_bnk_imax=0 --wall-treatment pinned --prior-wall-bc dirichlet)
# The g-adopt baseline: ROL Lin-More with L-BFGS, L2 inner product, free walls.
ROL=(--optimiser rol-lbfgs --riesz L2 --initial-radius 1.0)

# The arms. The required four, then the two optional ones.
T_ROL=()
T_N=(--hessian full --initial-radius 0.2)
T_N_R=(--hessian full --initial-radius 12.8)
T_GN_R=(--hessian gn --initial-radius 12.8)
T_GN=(--hessian gn --initial-radius 0.2)
T_GN_R_EW=(--hessian gn --initial-radius 12.8 --krylov-ew)
