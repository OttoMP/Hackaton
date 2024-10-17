# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as py
# %% Functions
# Function that returns the potential V in a given position x
def V(a):
    """a = x"""
    return (1.0/2.0)*a**2
# Hamiltonian function
def H(funcaoiii, funcaoii, funcaoi, potential, deltax):
    """funcaoiii = f(x_i+1,t_j),funcaoii = f(x_i,t_j),funcaoi = f(x_i-1,t_j)"""
    d = (funcaoiii-2.0*funcaoii+funcaoi)/(deltax**2.0)
    v = potential
    return -(1/2)*d + v*funcaoii
# Interaction GP Hamiltonian
def H_int(funcaoiii, funcaoii, funcaoi, potential, nonlinear, deltax):
    """funcaoiii = f(x_i+1,t_j),funcaoii = f(x_i,t_j),funcaoi = f(x_i-1,t_j), nonlinear = U_0"""
    d = (funcaoiii-2.0*funcaoii+funcaoi)/(deltax**2.0)
    v = potential
    u = nonlinear*(funcaoii**2)
    return -(1/2)*d + v*funcaoii + u*funcaoii
# Initial conditions to each |Ψ⟩
def psi_a(a):
    """t = 0, a = x"""
    return (1.0/np.pi)**(1.0/4.0)*np.exp(-a**2.0/2.0)
def psi_b(b):
    """t = 0, b = x"""
    return (2.0/np.pi)**(1.0/4.0)*np.exp(-b**2.0)
def psi_c(c):
    """t = 0, b = x"""
    return 1/(np.sqrt(2)*np.cosh(c))
# %% |Ψ_a⟩ and |Ψ_b⟩
# Variables definitions
L = 7.0 # This L was choosen due to 1/np.exp(7**2) = 5x10^-22 ~ 0
dx = 0.2 # choosing Δx arbitrarly
dt = 0.002#choose_dt(V(L),dx) # calculating Δt for the given Δx and maximum value of potential V(L)
print(-2/dt,'<', V(L), '<', 2/dt-2/(dx)**2, 'for |Ψ_a⟩ and |Ψ_b⟩') # Checking if the condition of Δt and Δx are satisfied
# Starting variables
x_ini = 0.0 # initial position
x = np.arange(-L, L, dx)
t_ini = 0.0; t_end = 10
t = np.arange(t_ini, t_end, dt/2.0)
# %%
# Start: R[i_x,j_t], I[i_x,j_t] and P[i_x,j_t] (i_x represents the index for position and i_t for time)
Ra = np.zeros((len(x),len(t))) # creating all vectors with the
Ia = np.zeros((len(x),len(t)))
Pa = np.zeros((len(x),len(t)))
Rb = np.zeros((len(x),len(t))) # creating all vectors with the
Ib = np.zeros((len(x),len(t)))
Pb = np.zeros((len(x),len(t)))

# %%
# Initial conditions for Psi_a and Psi_b for R(x,0) and I(x,Δt/2)
Ra[:,0] = np.real(psi_a(x))
for i in range(1,len(x)-1):
    Ia[i,1] = np.imag(psi_a(x[i]))-(dt/2.0)*H(Ra[i+1,0], Ra[i,0], Ra[i-1,0], V(x[i]), dx)
Rb[:,0] = np.real(psi_b(x))
for i in range(1,len(x)-1):
    Ib[i,1] = np.imag(psi_b(x[i]))-(dt/2.0)*H(Rb[i+1,0], Rb[i,0], Rb[i-1,0], V(x[i]), dx)
# %%
# We calculate R at t = n*Δt and I at t = (n+1/2)*Δt, for Psi_a and Psi_b
for j in range(1,len(t)-1):
    for i in range(1,len(x)-1):
        Ra[i,j+1] = Ra[i,j-1] + dt*(H(Ia[i+1,j],Ia[i,j],Ia[i-1,j],V(x[i]),dx))
    for i in range(1,len(x)-1):
        Ia[i,j+1] = Ia[i,j-1] - dt*(H(Ra[i+1,j],Ra[i,j],Ra[i-1,j],V(x[i]),dx))
for j in range(1,len(t)-1):
    for i in range(1,len(x)-1):
        Rb[i,j+1] = Rb[i,j-1] + dt*(H(Ib[i+1,j],Ib[i,j],Ib[i-1,j],V(x[i]),dx))
    for i in range(1,len(x)-1):
        Ib[i,j+1] = Ib[i,j-1] - dt*(H(Rb[i+1,j],Rb[i,j],Rb[i-1,j],V(x[i]),dx))
# %%
# Calculating the Probability Density
for j in range(1,len(t)-1):
    for i in range(len(x)):
        if j%2 == 0:
            Pa[i,j] = Ra[i,j]**2.0 + Ia[i,j+1]*Ia[i,j-1] # for an integer t[j]/t_end
        else:
            Pa[i,j] = Ia[i,j]**2.0 + Ra[i,j+1]*Ra[i,j-1] # for a half-integer t[j]/t_end
for j in range(1,len(t)-1):
    for i in range(len(x)):
        if j%2 == 0:
            Pb[i,j] = Rb[i,j]**2.0 + Ib[i,j+1]*Ib[i,j-1] # for an integer t[j]/t_end
        else:
            Pb[i,j] = Ib[i,j]**2.0 + Rb[i,j+1]*Rb[i,j-1] # for a half-integer t[j]/t_end
# %% 
# Plotting both initial conditions, |Ψ_a⟩ and |Ψ_b⟩, at same time
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
fotos = len(t)//100
a = py.confirm("Podemos começar a plotar?", buttons=["yes","no"])
if a == "yes":
    t_plot = np.arange(t_ini, t_end, dt)
    plot_adonai = np.arange(0, len(t_plot), fotos)
    for k in plot_adonai:
        name = 'foto'+str(k)+'.png'
        yname = r'$P(x,t/t_{total}=$'+str(round((t_plot[k]+dt)/t_end,1))+')'
        plt.figure(dpi=300)
        plt.plot(x, Pa[:,k],'b.:', label = r'$|\Psi_a\rangle$')
        plt.plot(x, Pb[:,k],'r.:', label = r'$|\Psi_b\rangle$')
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(yname, fontsize = 16)
        plt.ylim(-0.1,1.1)
        plt.grid(linestyle = ':')
        plt.title(r"Probability Density")
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig(name)
        plt.close()
else:
    print("Done")
# %% Interaction GPE, considering the non-linear term, |Ψ_c⟩
# Variables definitions
L = 7.0 # This L was choosen due to 1/np.exp(7**2) = 5x10^-22 ~ 0
V = 0 # potential
U_0 = -2 # nonlinear term
dx = 0.2 # Δx
dt = 0.0002# Δt
print(-2/dt,'<', V, '<', 2/dt-2/(dx)**2, 'for |Ψ_c⟩') # Checking if the condition of Δt and Δx are satisfied
# %%
# Starting variables
x_ini = 0.0 # initial position
x = np.arange(-L, L, dx)
t_ini = 0.0; t_end = 8
t = np.arange(t_ini, t_end, dt/2.0)
# %%
# Start: R[i_x,j_t], I[i_x,j_t] and P[i_x,j_t] (i_x represents the index for position and i_t for time)
Rc = np.zeros((len(x),len(t)))
Ic = np.zeros((len(x),len(t)))
Pc = np.zeros((len(x),len(t)))
# Initial conditions for Psi_c for R(x,0) and I(x,Δt/2)
Rc[:,0] = np.real(psi_c(x))
for i in range(1,len(x)-1):
    Ic[i,1] = np.imag(psi_c(x[i]))-(dt/2.0)*(H(Rc[i+1,0], Rc[i,0], Rc[i-1,0], V, dx)+U_0*(Rc[i,0]**2+np.imag(psi_c(x[i]))**2)*Rc[i,0])
# %%
# Calculating Re(Psi_c) and Im(Psi_c)
for j in range(1,len(t)-1):
    for i in range(1,len(x)-1):
        aux_r = 1.0 - dt*U_0*Rc[i,j-1]*Rc[i,j]
        Rc[i,j+1] = (Rc[i,j-1] + dt*(H_int(Ic[i+1,j],Ic[i,j],Ic[i-1,j],V,U_0,dx)))/(aux_r)
    for i in range(1,len(x)-1):
        aux_i = 1.0 + dt*U_0*Ic[i,j-1]*Ic[i,j]
        Ic[i,j+1] = (Ic[i,j-1] - dt*(H_int(Rc[i+1,j],Rc[i,j],Rc[i-1,j],V,U_0,dx)))/(aux_i)
# %%
# Calculating the Probability Density
for j in range(1,len(t)-2):
    for i in range(len(x)):
        if j%2 == 0:
            Pc[i,j] = Rc[i,j]**2.0 + Ic[i,j+1]*Ic[i,j-1] # for an integer t[j]/t_end
        else:
            Pc[i,j] = Ic[i,j]**2.0 + Rc[i,j+1]*Rc[i,j-1] # for a half-integer t[j]/t_end
# %%
fotos = len(t)//100 # number of pictures to be plotted for video
t_plot = np.arange(t_ini, t_end, dt)
plot_adonai = np.arange(0, len(t_plot), fotos)
import matplotlib.animation as animation

fig, ax = plt.subplots(dpi=300)
plt.xlabel(r'$x$', fontsize = 16)
plt.ylabel(r'$P\left(x,t\in [0,t_{end})\right)$', fontsize = 16)
plt.ylim(-0.1,1.1)
plt.grid(linestyle = ':')
plt.title(r"Probability Density")
plt.ylim(-0.1,1.1)
line, = ax.plot(x, Pc[:,0],'k.:', label = r'$|\Psi_c\rangle$')
plt.tight_layout()
plt.legend(loc='upper right')
def animate(i):
    line.set_ydata(Pc[:,i*fotos])  # update the data.
    print(i)
    return line,
ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, save_count=50)
plt.show()
# %% 
# Ploting |Ψ_c⟩
image_array = []
fotos = len(t)//100 # number of pictures to be plotted for video
c = py.confirm("Podemos começar a plotar |Ψ_c⟩?", buttons=["yes","no"])
if c == "yes":
    t_plot = np.arange(t_ini, t_end, dt)
    plot_adonai = np.arange(0, len(t_plot), fotos)
    for k in plot_adonai:
        name = 'foto'+str(k)+'.png'
        yname = r'$P(x,t/t_{total}=$'+str(round((t_plot[k]+dt)/t_end,1))+')'
        a = plt.figure(dpi=300)
        plt.plot(x, Pc[:,k],'k.:', label = r'$|\Psi_c\rangle$')
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(yname, fontsize = 16)
        plt.ylim(-0.1,1.1)
        plt.grid(linestyle = ':')
        plt.title(r"Probability Density")
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()
else:
    print("Done")