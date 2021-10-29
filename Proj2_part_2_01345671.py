import matplotlib.pyplot as plt
import numpy as np
from Proj2_part_1_01345671 import SOR


def gen_plot(rtol=1e-03, L=20, N=20, q=2, s=3, r=2, tau=0.05):
    title = "L * N = {0} * {1} to converge at tolerance {2}".format(L, N, rtol)
    ws = np.linspace(.5, 2, num=15)

    def get_iterations(w):
        scheme = SOR(np.zeros((L, N)), np.zeros((L, N)), q=q, s=s, r=r, tau=tau, w=w)
        _, iterations = scheme.iteration_loop(rtol=rtol, sweep_limit=1000)
        return iterations

    res = np.vectorize(get_iterations)(ws)

    fig = plt.figure()
    ax = fig.add_subplot(
        xlabel='omega', ylabel='iterations',
        title=title
    )
    ax.plot(ws, res, color='red', marker='.', linestyle='--')
    plt.savefig('./part_2_results/{}.png'.format(title))
    print('figure saved')


grid_schemes = [
    (20, 20),
    (50, 20),
    (50, 50)
]

if __name__ == "__main__":
    [gen_plot(L=L, N=N) for (L, N) in grid_schemes]
