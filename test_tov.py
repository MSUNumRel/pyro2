import numpy as np
import matplotlib.pyplot as plt

from gr.init_data import TOVInitialData
from gr import units
from hydro_base.eos import PolytropicEOS

from mesh import patch
from mesh.boundary import BC1d

# rho_c = units.convert(2e15, units.CGS.density, units.REL.density)
rho_c = 1.28e-3
rho_end = units.convert(1e12, units.CGS.density, units.REL.density)

r_c = units.convert(1e4, units.CGS.length, units.REL.length)
r_max = units.convert(15e5, units.CGS.length, units.REL.length)

eos = PolytropicEOS(100.0, 2.0)

grid = patch.Grid1d(100, 1, xmin=r_c, xmax=r_max)

print(grid.dx)

init_data = TOVInitialData(grid)

init_data.initialize(eos)

phi, r, m, P, rho0, eps = init_data.initial_guess(rho_c, rho_end)

print(rho_c, rho_end, r_c, r_max)
init_data.solve_to_grid(rho_c, rho_end)

bc = BC1d()
data = patch.CellCenterData1d(grid)

data.register_var("rho0", bc)
data.register_var("v", bc)
data.register_var("m", bc)
data.register_var("Phi", bc)
data.register_var("eps", bc)

data.create()

init_data.fill_patch(data)

plt.figure(1)

plt.semilogy(units.convert(grid.x[1:-1], units.REL.length, units.CGS.length)*1e-5,
         units.convert(data.get_var("rho0").v(), units.REL.density, units.CGS.density))
# plt.plot(units.convert(r, units.REL.length, units.CGS.length)*1e-5,
#          units.convert(rho0, units.REL.density, units.CGS.density))

# plt.plot(grid.x[1:-1], data.get_var("rho0").v())

plt.show()
