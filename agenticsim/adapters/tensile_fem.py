"""Small-strain triangular plane-strain FEM; numerical libraries load on demand."""
from agenticsim.domain import require


class PlaneStrainMesh:
    def __init__(self, xy, tri, boundaries=None):
        import numpy as np
        from scipy.sparse import coo_matrix
        self.np, self.coo = np, coo_matrix
        self.xy, self.tri = np.asarray(xy, float), np.asarray(tri, int)
        require(self.xy.ndim == 2 and self.xy.shape[1] == 2 and self.tri.ndim == 2 and self.tri.shape[1] == 3,
                'expected 2D TRI3 mesh')
        require(self.tri.min() >= 0 and self.tri.max() < len(self.xy), 'triangle indices outside mesh')
        points = self.xy[self.tri]
        determinant = ((points[:, 1, 0]-points[:, 0, 0])*(points[:, 2, 1]-points[:, 0, 1])
                      -(points[:, 2, 0]-points[:, 0, 0])*(points[:, 1, 1]-points[:, 0, 1]))
        require(np.isfinite(points).all() and np.all(np.abs(determinant) > 1e-14), 'degenerate/nonfinite triangles')
        self.area = np.abs(determinant)/2
        dx = np.column_stack((points[:, 1, 1]-points[:, 2, 1], points[:, 2, 1]-points[:, 0, 1],
                              points[:, 0, 1]-points[:, 1, 1]))/determinant[:, None]
        dy = np.column_stack((points[:, 2, 0]-points[:, 1, 0], points[:, 0, 0]-points[:, 2, 0],
                              points[:, 1, 0]-points[:, 0, 0]))/determinant[:, None]
        self.b = np.zeros((len(tri), 4, 6))
        self.b[:, 0, 0::2] = dx
        self.b[:, 1, 1::2] = dy
        self.b[:, 3, 0::2] = dy
        self.b[:, 3, 1::2] = dx
        self.dofs = (2*self.tri[:, :, None] + np.arange(2)).reshape(-1, 6)
        self.rows = np.repeat(self.dofs, 6, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 6)).ravel()
        self.bottom = (np.asarray(boundaries['bottom'], int) if boundaries else
                       np.flatnonzero(np.isclose(self.xy[:, 1], self.xy[:, 1].min(), atol=1e-10, rtol=0)))
        self.top = (np.asarray(boundaries['top'], int) if boundaries else
                    np.flatnonzero(np.isclose(self.xy[:, 1], self.xy[:, 1].max(), atol=1e-10, rtol=0)))
        for indices in (self.bottom, self.top):
            require(len(indices) >= 2 and indices.min() >= 0 and indices.max() < len(xy), 'invalid boundary nodes')
        self.top = self.top[np.argsort(self.xy[self.top, 0])]
        left = int(boundaries['bottom_left'][0]) if boundaries else self.bottom[np.argmin(self.xy[self.bottom, 0])]
        require(left in self.bottom and np.ptp(self.xy[self.top, 1]) < 1e-10, 'unsupported pinned/curved top boundary')
        self.fixed = np.r_[2*self.bottom+1, 2*left]
        self.free = np.setdiff1d(np.arange(2*len(xy)), self.fixed)
        self.node_area = np.bincount(self.tri.ravel(), weights=np.repeat(self.area, 3), minlength=len(xy))
        self.node_count = np.bincount(self.tri.ravel(), minlength=len(xy))
        require(len(self.top) >= 2 and np.all(self.node_area > 0), 'missing boundary or unused nodes')

    def elasticity(self, young, poisson):
        np = self.np
        require(young > 0 and 0 < poisson < 0.49, 'invalid elasticity constants')
        mu = young/(2*(1+poisson))
        lam = young*poisson/((1+poisson)*(1-2*poisson))
        d = np.zeros((4, 4))
        d[:3, :3] = lam
        d[np.arange(3), np.arange(3)] += 2*mu
        d[3, 3] = mu
        return d

    def force(self, pressure):
        np = self.np
        force = np.zeros(2*len(self.xy))
        lengths = np.linalg.norm(np.diff(self.xy[self.top], axis=0), axis=1)
        np.add.at(force, 2*self.top[:-1]+1, -pressure*lengths/2)
        np.add.at(force, 2*self.top[1:]+1, -pressure*lengths/2)
        return force

    def assemble(self, tangent):
        np = self.np
        element = np.einsum('eai,eab,ebj,e->eij', self.b, tangent, self.b, self.area, optimize=True)
        return self.coo((element.ravel(), (self.rows, self.cols)), shape=(2*len(self.xy),)*2).tocsr()

    def internal(self, stress):
        np = self.np
        forces = np.einsum('eai,ea,e->ei', self.b, stress, self.area, optimize=True)
        return np.bincount(self.dofs.ravel(), weights=forces.ravel(), minlength=2*len(self.xy))

    def recover(self, element, method='uniform'):
        np = self.np
        require(method in ('uniform', 'area'), 'unsupported nodal stress recovery')
        weight = self.area if method == 'area' else np.ones(len(self.tri))
        denominator = self.node_area if method == 'area' else self.node_count
        return np.column_stack([np.bincount(self.tri.ravel(), weights=np.repeat(element[:, i]*weight, 3),
                                           minlength=len(self.xy))/denominator for i in range(element.shape[1])])

    def elastic(self, young, poisson, pressure):
        np = self.np
        from scipy.sparse.linalg import spsolve
        d = self.elasticity(young, poisson)
        stiffness = self.assemble(np.broadcast_to(d, (len(self.tri), 4, 4)))
        force = self.force(pressure)
        displacement = np.zeros(2*len(self.xy))
        displacement[self.free] = spsolve(stiffness[self.free][:, self.free], force[self.free])
        strain = np.einsum('eai,ei->ea', self.b, displacement[self.dofs])
        stress = strain @ d.T
        residual = self.internal(stress)-force
        relative = np.linalg.norm(residual[self.free])/max(np.linalg.norm(force[self.free]), 1e-30)
        return self.outputs(displacement, stress), {'relative_equilibrium_residual': float(relative),
                                                   'bc_max_abs': float(np.abs(displacement[self.fixed]).max()),
                                                   'converged': bool(relative < 1e-8)}

    def outputs(self, displacement, stress):
        np = self.np
        nodal = self.recover(stress)
        u = displacement.reshape(-1, 2)
        fields = np.column_stack((u[:, 0], u[:, 1], nodal[:, 0], nodal[:, 1], nodal[:, 3]))
        # The dataset QoI is the in-plane convention, not the 3D plane-strain invariant.
        von_mises = np.sqrt(nodal[:, 0]**2 + nodal[:, 1]**2 - nodal[:, 0]*nodal[:, 1] + 3*nodal[:, 3]**2)
        scalars = np.array([von_mises.max(), u[self.top, 1].max(), nodal[self.top, 1].max()])
        require(np.isfinite(fields).all() and np.isfinite(scalars).all(), 'nonfinite FEM solution')
        return fields, scalars
