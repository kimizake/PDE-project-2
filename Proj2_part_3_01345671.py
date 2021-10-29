import matplotlib.pyplot as plt
import numpy as np
from Proj2_part_1_01345671 import get_neighbours, SOR


class ModifiedSOR(SOR):

    def get_value(self, phi, i, j):
        x = i * self.dx - self.q
        y = j * self.dy

        ns = get_neighbours(phi, i, j)

        """
            Require U_-1 = U_1 to apply Neumann boundary conditions
        """
        rows, cols = phi.shape
        if j == cols - 1:
            ns.update({'n': phi[i, j - 1]})
        if j == 0:
            ns.update({'s': phi[i, j + 1]})
        if i == 0:
            ns.update({'w': phi[i + 1, j]})
        if i == rows - 1:
            ns.update({'e': phi[i - 1, j]})

        if 0 < x < 0.5 and 0.5 < x < 1:
            """
                In this range the transformation has been applied, so we treat
                y values as eta
            """
            y_b_x = 2 * self.tau * (1 - 2 * x)
            m = (1 - 1 / (y_b_x ** 2)) / (self.dx ** 2)
            n = 1 / (self.dy ** 2)
            return (m * (ns['e'] + ns['w']) +
                    n * (ns['n'] + ns['s']) -
                    self.fs[i, j]) / (2 * m * n)
        else:
            return self.step_func(phi, ns, self.fs[i, j])


if __name__ == "__main__":
    """
        Must have L and N of form 2^p + 1
    """
    q, s, r, L, N, tao, w = 2, 3, 1, 65, 65, 0.05, 1.8
    scheme = ModifiedSOR(
        np.zeros((L, N)), np.zeros((L, N)),
        q=q, s=s, r=r, tao=tao, w=w
    )
    u, step = scheme.iteration_loop(sweep_limit=100000)
    # us, vs = scheme.get_derivatives()
    X = np.linspace(-q, s, num=L)
    Y = np.linspace(0, r, num=N)
    [us, vs] = np.gradient(u)

    field = plt.figure(1)
    ax1 = field.add_subplot()
    q = ax1.quiver(X, Y, np.transpose(us), np.transpose(vs), np.transpose(u),
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
    ax3 = u_surface.add_subplot(projection='3d')
    ax3.plot_surface(Xm, Ym, np.transpose(us), cmap='plasma')
    ax3.set_title('U_surface')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    v_surface = plt.figure(4)
    ax3 = v_surface.add_subplot(projection='3d')
    ax3.plot_surface(Xm, Ym, np.transpose(vs), cmap='plasma')
    ax3.set_title('V_surface')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    plt.show()
