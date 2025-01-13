# convergence plots as in Fraters et al. (2019) for Spiegelman et al. (2019) case
# run with a variable number of initial Picard iterations and with or without Jacobian stabilisation
import os.path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.formatter.use_mathtext'] = True

# Set up geometry:
nx, ny = 64, 16

mu0 = 1e22


def spiegelman(U0, mu1, nx, ny, picard_iterations, stabilisation=False):
    output_dir = f"spiegelman_{float(U0)}_{float(mu1*mu0)}_{nx}_{ny}_{picard_iterations}_{stabilisation}"
    residuals = [float(line[4:]) for line in open(os.path.join(output_dir, 'picard.txt'))]
    residuals += [float(line[23:]) for line in open(os.path.join(output_dir, 'newton.txt'))]
    return np.array(residuals)


for ui, mui, mu_latex in zip([2.5e-3, 5e-3, 12.5e-3], [1e23, 1e24, 5e24], ['10^{23}', '10^{24}', r'5\times 10^{24}']):
    plt.figure()
    for picard_iterations, color in zip([50, 0, 5, 15, 25], ['tab:purple', 'k', 'b', 'tab:orange', 'r']):
        for stab in [False, True]:
            if stab and picard_iterations == 50:
                break
            residuals = spiegelman(ui, mui/mu0, nx, ny, picard_iterations, stabilisation=stab)
            if picard_iterations == 50:
                label = 'Picard'
                marker = '-'
            elif stab:
                label = f'{picard_iterations} Picard + stab. Newton'
                marker = '.'
            else:
                label = f'{picard_iterations} Picard + unst. Newton'
                marker = '-'
            plt.semilogy(residuals/residuals[0], marker, label=label, color=color)

    plt.axis([0, 50, 1e-14, 1])
    plt.legend()
    ax = plt.gca()
    ax.set_title(rf'$U_0={ui*1000}$ mm/yr, $\eta_1={mu_latex}$ Pa$\,$s')
    plt.xlabel('Picard/Newton Iterations')
    plt.ylabel(r'Residual $\alpha$')
    plt.savefig(f'spiegelman_{nx}_{ny}_{ui}_{mui}.pdf')
