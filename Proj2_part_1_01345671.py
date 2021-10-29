import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_neighbours(m, i, j):
    rows, cols = m.shape
    out = {
        'n': 0,
        'ne': 0,
        'e': 0,
        'se': 0,
        's': 0,
        'sw': 0,
        'w': 0,
        'nw': 0,
        'c': m[i, j]
    }
    if j < cols - 1:
        out.update({'n': m[i, j + 1]})
    if j > 0:
        out.update({'s': m[i, j - 1]})
    if i < rows - 1:
        out.update({'e': m[i + 1, j]})
    if i > 0:
        out.update({'w': m[i - 1, j]})
    if j < cols - 1 and i < rows - 1:
        out.update({'ne': m[i + 1, j + 1]})
    if j > 0 and i < rows - 1:
        out.update({'se': m[i + 1, j - 1]})
    if j < cols - 1 and i > 0:
        out.update({'nw': m[i - 1, j + 1]})
    if j > 0 and i > 0:
        out.update({'sw': m[i - 1, j - 1]})
    return out


class Jacobi(object):

    def __init__(self, u_in, fs, **kwargs):
        self.q = kwargs['q']
        self.s = kwargs['s']
        self.r = kwargs['r']
        self.L, self.N = u_in.shape
        self.tau = kwargs['tau']

        self.dx = (self.s + self.q) / (self.L - 1)
        self.dy = self.r / (self.N - 1)

        self.u = u_in
        self.fs = fs

    def apply_boundary_conditions_to_neighbours(self, phi, i, j):
        """
        applies the boundary conditions to neigbours of phi[i, j]
        :param phi: matrix of unknowns for the next iteration step
        :param i: x-index
        :param j: y-index
        :return: dict of neaighbours around point phi[i, j]
        """
        x = i * self.dx - self.q
        y = j * self.dy

        ns = get_neighbours(phi, i, j)

        rows, cols = phi.shape
        """
           Require U_-1 = U_1 to apply Neumann boundary conditions
        """
        if j == cols - 1:
            ns.update({'n': phi[i, j - 1]})
        if j == 0:
            ns.update({'s': phi[i, j + 1]})
        if i == 0:
            ns.update({'w': phi[i + 1, j]})
        if i == rows - 1:
            ns.update({'e': phi[i - 1, j]})

        if y == 0 and 0 <= x <= 1:
            y_b_x = 2 * self.tau * (1 - 2 * x)
            cd_x = (ns['e'] - ns['w']) / (2 * self.dx)
            ns.update(
                {'s': (ns['n'] - 2 * self.dy * y_b_x * (1 + cd_x))}
            )

        return ns

    def step_func(self, ns, f):
        m = 1 / (self.dx ** 2)
        n = 1 / (self.dy ** 2)
        return (m * (ns['w'] + ns['e']) +
                n * (ns['n'] + ns['s']) -
                f) / (2 * (m + n))

    def residual(self, ns, f):
        m = 1 / (self.dx ** 2)
        n = 1 / (self.dy ** 2)
        return f - (
            m * (ns['w'] + ns['e']) +
            n * (ns['n'] + ns['s']) -
            2 * (m + n) * ns['c']
        )

    def iteration_step(self, curr, _, i, j):
        """
        This method is used to distinguish between the Jacobi scheme and the G-S scheme.
        Since their code is largely the same, the only difference is this method which acts as a
        'switch' for the dynamic programming.
        :param curr: matrix u at iteration j
        :param _: matrix u at iteration j+1 (the one being updated)
        :param i: x-index
        :param j: y-index
        :return: u[i, j] at level j+1
        """
        ns = self.apply_boundary_conditions_to_neighbours(curr, i, j)
        f = self.fs[i, j]
        return self.step_func(ns, f)

    def compute_residuals(self, phi, fs):
        L, N = phi.shape
        u_out = np.zeros((L, N))
        for i in range(L):
            for j in range(N):
                ns = self.apply_boundary_conditions_to_neighbours(phi, i, j)
                f = fs[i, j]
                u_out[i, j] = self.residual(ns, f)
        return u_out

    def iteration_loop(self, rtol=1e-01, sweep_limit=1000):
        sweep = 1  # counter
        while True:
            """
                perform each iteration step until convergence
                u is the array we use in the next iteration
                self.U is our current
            """
            curr = self.u
            next = np.copy(curr)  # copy by value
            for i in range(self.L):
                for j in range(self.N):
                    next[i, j] = self.iteration_step(curr, next, i, j)

            """
                compute the residuals of the next u
            """
            res = self.compute_residuals(next, self.fs)
            if np.linalg.norm(res) < rtol:
                """
                    check for convergence by measuring the euclidian distance of the residue from the origin.
                """
                break
            elif sweep > sweep_limit:
                """
                    want to break in case convergence is taking too long
                """
                break
            else:
                """
                    else we update our current matrix of unknowns for the next iteration
                    and increment counter
                """
                self.u = next
                sweep = sweep + 1
        return self.u, sweep


class GS(Jacobi):

    def iteration_step(self, _, _next, i, j):
        ns = self.apply_boundary_conditions_to_neighbours(_next, i, j)
        f = self.fs[i, j]
        return self.step_func(ns, f)


class SOR(GS):

    def __init__(self, u_in, f, **kwargs):
        super().__init__(u_in, f, **kwargs)
        self.w = kwargs.get('w')

    def step_func(self, ns, f):
        m = 1 / (self.dx ** 2)
        n = 1 / (self.dy ** 2)
        return (1 - self.w) * ns['c'] + self.w * (
                m * (ns['w'] + ns['e']) +
                n * (ns['n'] + ns['s']) -
                f) / (2 * (m + n))


if __name__ == "__main__":
    q, s, r, L, N, tau, w = 2, 3, 2, 51, 21, 0.05, 1.7
    scheme = SOR(
        np.zeros((L, N)), np.zeros((L, N)),
        q=q, s=s, r=r, tau=tau, w=w
    )
    u, step = scheme.iteration_loop(rtol=1e-02, sweep_limit=3000)
    # us, vs = scheme.get_derivatives()
    X = np.linspace(-q, s, num=L)
    Y = np.linspace(0, r, num=N)
    [us, vs] = np.gradient(u)
    print(step)

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

    # xs = [0.025, 0.25, 0.5, 0.75, 0.95]
    # dx = (s + q) / (L - 1)
    # _is = [(x + q) / dx for x in xs]
    # [print(us[int(i)][0]) for i in np.rint(_is)]

    plt.show()

