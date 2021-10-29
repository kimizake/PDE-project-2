from Proj2_part_1_01345671 import get_neighbours, GS
import numpy as np
import matplotlib.pyplot as plt


def restrict(fine):
    """

    :param fine: nd.array of shape (2L, 2N)
    :return: nd.array of shape (L, N)
    """
    rows, cols = fine.shape
    L = int((rows - 1) / 2) + 1
    N = int((cols - 1) / 2) + 1
    u_out = np.zeros((L, N))
    for j in range(N):
        for i in range(L):
            ns = get_neighbours(fine, 2 * i, 2 * j)
            u_out[i, j] = (1 / 4) * (ns['c']) + \
                          (1 / 8) * (ns['n'] + ns['e'] + ns['s'] + ns['w']) + \
                          (1 / 16) * (ns['ne'] + ns['se'] + ns['sw'] + ns['nw'])
    return u_out


def interpolate(coarse):
    """

    :param coarse: nd.array of shape(L, N)
    :return: nd.array of shape (2L, 2N)
    """
    L, N = coarse.shape
    u_out = np.zeros((2 * (L - 1) + 1, 2 * (N - 1) + 1))
    for j in range(N):
        for i in range(L):
            ns = get_neighbours(coarse, i, j)
            u_out[2 * i, 2 * j] = ns['c']
            try:
                u_out[2 * i + 1, 2 * j] = (1 / 2) * (ns['c'] + ns['e'])
            except IndexError:
                pass
            try:
                u_out[2 * i, 2 * j + 1] = (1 / 2) * (ns['c'] + ns['n'])
            except IndexError:
                pass
            try:
                u_out[2 * i + 1, 2 * j + 1] = (1 / 4) * (ns['c'] + ns['e'] + ns['n'] + ns['ne'])
            except IndexError:
                pass
    return u_out


def multi_grid(u_in, f, **kwargs):
    """

    :param u_in: nd.array of shape (2N, 2N)
    :param f: nd.array of shape (2L, 2N)
    :param kwargs: params for iteration scheme
    :return: smoothed out u_in
    """
    m, n = f.shape
    gs = GS(u_in, f, **kwargs)
    u_f, _ = gs.iteration_loop(rtol=1e-03, sweep_limit=10)

    if n <= 3 or m <= 3:
        return u_f
    else:
        r_f = gs.compute_residuals(u_f, f)
        r_c = restrict(r_f)
        z_c = multi_grid(np.zeros(r_c.shape), r_c, **kwargs)
        z_f = interpolate(z_c)
        u_f = u_f.__add__(z_f)
        _gs = GS(u_f, f, **kwargs)
        u_out, _ = _gs.iteration_loop(rtol=1e-03, sweep_limit=10)
        return u_out


if __name__ == "__main__":
    """
        Must have L and N of form 2^p + 1
    """
    import time

    start_time = time.time()
    q, s, r, L, N, tau, w = 2, 3, 1, 257, 257, 0.05, 1.2
    u = multi_grid(
        np.zeros((L, N)), np.zeros((L, N)),
        q=q, s=s, r=r, tau=tau, w=w
    )
    # us, vs = scheme.get_derivatives()
    X = np.linspace(-q, s, num=L)
    Y = np.linspace(0, r, num=N)
    [us, vs] = np.gradient(u)

    field = plt.figure(1)
    ax1 = field.add_subplot()
    ax1.quiver(X, Y, np.transpose(us), np.transpose(vs), np.transpose(u),
               cmap='plasma')
    ax1.set_title('Vector field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    surface = plt.figure(2)
    ax2 = surface.add_subplot(projection='3d')
    Xm, Ym = np.meshgrid(X, Y)
    ax2.plot_surface(Xm, Ym, np.transpose(u), cmap='plasma')
    ax2.set_title('Surface')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    u_surface = plt.figure(3)
    ax3 = u_surface.add_subplot(title='U Surface at y=0', xlabel='x', ylabel='U')
    ax3.plot(X, np.transpose(us)[0], color='red', marker='.', linestyle='--')

    v_surface = plt.figure(4)
    ax4 = v_surface.add_subplot(title='V Surface at y=0', xlabel='x', ylabel='v')
    ax4.plot(X, np.transpose(vs)[0], color='red', marker='.', linestyle='--')

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
