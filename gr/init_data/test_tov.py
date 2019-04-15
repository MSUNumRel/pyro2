import numpy as np

from gr.init_data import TOVInitialData
from gr import units
from hydro_base.eos import PolytropicEOS

from mesh import patch

rho_c = units.convert(2e15, units.CGS.density, units.REL.density)
rho_end = units.convert(1e0, units.CGS.density, units.REL.density)

r_c = units.convert(1e2, units.CGS.length, units.REL.length)
r_max = units.convert(20e5, units.CGS.length, units.REL.length)

eos = PolytropicEOS(3e4, 2.75)

grid = patch.Grid1d(1000, 1, xmin=r_c, xmax=r_max)

init_data = TOVInitialData(grid)

init_data.initialize(eos)

U = init_data.initial_guess(rho_c, rho_end)

print(U)
