import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from ufl import TrialFunction, TestFunction, dx, grad, inner
from dolfinx import mesh, fem
from dolfinx.mesh import locate_entities_boundary, meshtags
import math
import matplotlib.animation as animation
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

# --- Create 1D mesh ---
neg_x= -1.
pos_x= 1.
domain = mesh.create_interval(MPI.COMM_WORLD, 200, [neg_x, pos_x])  # Domain from -1.0 to 1.0

# --- Create function space ---
import basix.ufl
element = basix.ufl.element("CG", basix.ElementFamily.P, 1)
V = fem.functionspace(domain, element)

# --- Constants ---
D0_SS = 2e-5          # Initial Diffusion coefficient 
D0_in = 1.1e-6      # Initial Diffusion coefficient
Q_ss = 157          # activation energy (J/mol)
Q_inconel = 183               # activation energy (J/mol)
Temp = 1060 + 273     # absolute temperature (K)
R = 8.314          # gas constant (8.314 J/mol·K)
D_SS = D0_SS * math.exp(-Q_ss / (R * Temp))
D_inconel = D0_in * math.exp(-Q_inconel / (R * Temp))
u0_ss = 0.0066 # initial carbon concentration ss316 mol
u0_inconel = 0.066 # initial carbon concentration inconel mol
T = 3600           # Total time
num_steps = 36000     # Number of time steps
dt = T / num_steps

# # --- Smooth transition for diffusion coefficient across boundary ---
# def smooth_transition(x, D1, D2, interface_position=0, width=0.1):
#     """
#     Sigmoid function for smooth transition between two materials' diffusion coefficients.
#     """
#     return D1 + (D2 - D1) / (1 + np.exp(-100 * (x - interface_position) / width))

# # --- Define diffusion coefficient D(x) as smooth transition ---
# DG0 = fem.functionspace(domain, ("DG", 0))
# D_func = fem.Function(DG0)

# # Apply smooth transition across the domain
# D_array = smooth_transition(domain.geometry.x[:, 0], D_SS, D_inconel)  # Smooth transition for diffusion coefficient

# local_start, local_end = domain.topology.index_map(domain.topology.dim).local_range
# D_array_local = D_array[local_start:local_end]

DG0 = fem.functionspace(domain, ("DG", 0))  # Discontinuous Galerkin space for piecewise constant functions
D_func = fem.Function(DG0)

# Get x-coordinates of cell centers
x = domain.geometry.x[:, 0]
local_start, local_end = domain.topology.index_map(domain.topology.dim).local_range
x_local = x[local_start:local_end]

# Assign piecewise constant D
D_array_local = np.where(x_local < 0, D_SS, D_inconel)
D_func.x.array[:] = D_array_local


# --- Neumann boundary fluxes --- 
fdim = domain.topology.dim - 1

def left_boundary(x): return np.isclose(x[0], -1.0)
def right_boundary(x): return np.isclose(x[0], 1.0)

left_facets = locate_entities_boundary(domain, fdim, left_boundary)
right_facets = locate_entities_boundary(domain, fdim, right_boundary)

all_facets = np.hstack([left_facets, right_facets])
all_values = np.hstack([np.full_like(left_facets, 2), np.full_like(right_facets, 3)])

facet_tags = meshtags(domain, fdim, all_facets, all_values)

# --- Mark subdomains for materials ---
tdim = domain.topology.dim
cell_midpoints = domain.geometry.x.mean(axis=1)
material_ids = np.where(cell_midpoints < 0, 1, 2)  # Material 1  on the left, Material 2  on the right
material_markers = meshtags(domain, tdim, np.arange(len(material_ids), dtype=np.int32), material_ids)

# --- Define measures ---
dx = ufl.Measure("dx", domain=domain, subdomain_data=material_markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


# --- Trial and Test functions ---
u = TrialFunction(V)
v = TestFunction(V)

# --- Previous time step function ---
u_n = fem.Function(V)
# Initial concentration: set piecewise initial values
u_n_vals = np.zeros_like(material_ids, dtype=np.float64)
for mat_id, val in enumerate([u0_ss, u0_inconel], start=1):
    u_n_vals[material_ids == mat_id] = val
# Interpolate initial values to CG1 space
x = domain.geometry.x[:, 0]
u_n_array = np.zeros_like(x)
u_n_array[x < 0] = u0_ss
u_n_array[x >= 0] = u0_inconel
u_n.x.array[:] = u_n_array
u_n.x.scatter_forward()

# --- Neumann flux constants (flux in mol/(m^2 s)) --- 
flux_left = fem.Constant(domain, PETSc.ScalarType(0.0))  # Zero flux at left boundary
flux_right = fem.Constant(domain, PETSc.ScalarType(0.0))  # Zero flux at right boundary

# --- Variational formulation (Backward Euler) ---
a = u * v * dx + dt * inner(D_func * grad(u), grad(v)) * dx
L = u_n * v * dx + dt * flux_left * v * ds(2) + dt * flux_right * v * ds(3)

# --- Assemble system ---
uh = fem.Function(V)

# Create linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.JACOBI)

# --- Prepare to store solutions for animation ---
all_u_vals = []
all_diff_lengths = []
all_times = []

for n in range(num_steps):
    # Assemble system matrix and RHS
    A = assemble_matrix(fem.form(a))
    A.assemble()
    b = assemble_vector(fem.form(L))

    # Solve linear system
    solver.setOperators(A)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update previous solution
    u_n.x.array[:] = uh.x.array

    # Store for animation
    u_current = uh.x.array.copy()
    all_u_vals.append(u_current)

    # Calculate diffusion length at current step
    threshold = 0.1 * u_current.max()
    indices = np.where(u_current >= threshold)[0]
    if len(indices) > 0:
        diff_length = x[indices[-1]] - x[indices[0]]
    else:
        diff_length = 0.0
    all_diff_lengths.append(diff_length)
    all_times.append(dt * (n+1))

# --- Animation using matplotlib ---
print( "initial animation ... ")
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], 'b-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
length_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, max(np.max(u) for u in all_u_vals)*1.1)
ax.set_xlabel('Position (m)')
ax.set_ylabel('Carbon Concentration')
ax.set_title('Carbon Diffusion in 1D Domain')

# Visualize material boundary at x=0 (no sharp line)
ax.axvline(0, color='gray', linestyle='--', label='Material interface')
ax.axvspan(-1.0, 0, color='lightblue', alpha=0.2, label='SS316')
ax.axvspan(0, 1.0, color='orange', alpha=0.2, label='Inconel')

def init():
    line.set_data([], [])
    time_text.set_text('')
    length_text.set_text('')
    return line, time_text, length_text

def animate(i):
    y = all_u_vals[i]
    line.set_data(x, y)
    time_text.set_text(f'Time = {all_times[i]:.1f} s')
    length_text.set_text(f'Diffusion length = {all_diff_lengths[i]:.5f} m')
    return line, time_text, length_text
print( 'take a while')
ani = animation.FuncAnimation(fig, animate, frames=num_steps,
                              init_func=init, blit=True, interval=100)

# Save as mp4
print('saving')
ani.save('diffusion_animation12.mp4', writer='ffmpeg', fps=10)
# Or save as gif (uncomment if you prefer)
# ani.save('diffusion_animation.gif', writer='imagemagick', fps=10)
print('finish')
plt.show()
