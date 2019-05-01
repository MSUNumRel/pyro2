# simple inputs files for the SR sod shock tube

[driver]
max_steps = 2000
tmax = 10.0

[io]
basename = gr_os
dt_out = 0.05

[mesh]
nx = 500
xmin = 1e3
xmax = 15e6
xlboundary = outflow
xrboundary = outflow

[os]
radMass_ratio = 5.0
mass = 5.0

[eos]
gamma = 1.0
K = 0

[gr_radial_solver]
riemann = HLLE

[compressible_GR]
riemann = HLLE

[gr]
evolve_metric = 0
