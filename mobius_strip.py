import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz


class MobiusStrip:
    def __init__(self, R=1, w=0.2, n=100):
        self.R = R  # Radius
        self.w = w  # Width
        self.n = n  # Resolution

        # Create u and v ranges
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Compute surface points
        self.X, self.Y, self.Z = self.parametric_surface()

    def parametric_surface(self):
        u = self.U
        v = self.V
        R = self.R

        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        return x, y, z

    def compute_surface_area(self):
        # Compute partial derivatives
        dXdu, dXdv = np.gradient(self.X)
        dYdu, dYdv = np.gradient(self.Y)
        dZdu, dZdv = np.gradient(self.Z)

        # Cross product of partial derivatives
        cross_X = dYdu * dZdv - dZdu * dYdv
        cross_Y = dZdu * dXdv - dXdu * dZdv
        cross_Z = dXdu * dYdv - dYdu * dXdv

        dA = np.sqrt(cross_X**2 + cross_Y**2 + cross_Z**2)

        # Double integration over surface
        area = trapz(trapz(dA, self.v, axis=0), self.u)
        return area

    def compute_edge_length(self):
        # Edge at v = w/2
        R = self.R
        w = self.w
        u = self.u
        v = w / 2
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)

        length = trapz(ds, u)
        return length

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', linewidth=0.1)
        ax.set_title('Möbius Strip')
        plt.tight_layout()
        plt.show()

# === Main Execution ===
if __name__ == "__main__":
    strip = MobiusStrip(R=1, w=0.4, n=200)
    area = strip.compute_surface_area()
    length = strip.compute_edge_length()

    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {length:.4f}")

    strip.plot()
