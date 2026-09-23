import math

gamma = 1.4
rho_in = 1.0
p_in = 0.85
R_in = 0.5
throat_ratio = 0.2
mach_out = 3.5


A_in = math.pi * R_in**2
A_throat = math.pi * (throat_ratio * R_in) ** 2

c_in = (gamma * p_in / rho_in) ** 0.5
mach_in = 0.5
for i in range(100):
    mach_in = (
        A_throat
        / A_in
        * ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * mach_in**2))
        ** ((gamma + 1) / (2 * (gamma - 1)))
    )
mach_in = (
    A_throat
    / A_in
    * ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * mach_in**2))
    ** ((gamma + 1) / (2 * (gamma - 1)))
)
u_in = mach_in * c_in
massflow_in = rho_in * u_in * A_in
rho0 = rho_in * (1 + (gamma - 1) / 2 * mach_in**2) ** (1 / (gamma - 1))
p0 = p_in * (1 + (gamma - 1) / 2 * mach_in**2) ** (gamma / (gamma - 1))
rho_throat = rho0 * (2 / (gamma + 1)) ** (1 / (gamma - 1))
p_throat = p0 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
c_throat = (gamma * p_throat / rho_throat) ** 0.5
A_out = (
    A_throat
    / mach_out
    * ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * mach_out**2))
    ** ((gamma + 1) / (2 * (gamma - 1)))
)
R_out = (A_out / math.pi) ** 0.5
rho_out = rho0 * (1 + (gamma - 1) / 2 * mach_out**2) ** (-1 / (gamma - 1))
p_out = p0 * (1 + (gamma - 1) / 2 * mach_out**2) ** (-gamma / (gamma - 1))

c_out = (gamma * p_out / rho_out) ** 0.5
u_out = mach_out * c_out

print(
    f"rho_in: {rho_in:.6f}, p_in: {p_in:.6f}, R_in: {R_in:.6f}, u_in: {u_in:.6f}, c_in: {c_in:.6f}, mach_in: {mach_in:.6f}"
)
print(f"rho0: {rho0:.6f}, p0: {p0:.6f}")
print(
    f"rho_throat: {rho_throat:.6f}, p_throat: {p_throat:.6f}, c_throat: {c_throat:.6f}"
)
print(
    f"rho_out: {rho_out:.6f}, p_out: {p_out:.6f}, c_out: {c_out:.6f}, u_out: {u_out:.6f}"
)
print(f"R_out: {R_out:.6f}, A_out: {A_out:.6f}")
print(f"massflow_in: {massflow_in:.6f}")
