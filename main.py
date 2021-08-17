
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym


wc = 5.0 * 2 * np.pi  # cavity frequency
wa = 5.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.00  # cavity dissipation rate
gamma = 0.0  # atom dissipation rate
N = 2  # number of cavity fock states
n_th_a = 0.0  # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 6, 100)

wc1 = 4.979 * 2 * np.pi  # cavity frequency1
wc2 = 4.989 * 2 * np.pi  # cavity frequency2
wc3 = 5.004 * 2 * np.pi
wc4 = 5.024 *2 * np.pi
wc5 = 5.046 *2 * np.pi
wc6 = 5.068 *2 * np.pi
wc7 = 5.088 *2 * np.pi
wc8 = 5.105 *2 * np.pi
wc9 = 5.116 *2 * np.pi
wa = 5.0 * 2 * np.pi  # atom frequency
p=1e-3
g1 = 5.0 * 2 * np.pi * p  # coupling strength
g2 = 68.4375 * 2 * np.pi * p # coupling strength
g3 = 113.75 * 2 * np.pi
g4 = 140.938 * 2 * np.pi
g5 = 150.0 * 2 * np.pi
g6 = 140.938 * 2 * np.pi
g7 = 113.75 * 2 * np.pi
g8 = 68.4375 * 2 * np.pi
g9 = 5.0 * 2 * np.pi
kappa = 0.00  # cavity dissipation rate
gamma = 0.0  # atom dissipation rate
N = 2  # number of cavity fock states
n_th_a = 0.0  # temperature in frequency units

tlist = np.linspace(0, 6, 100)

# intial state
psi0 = tensor(basis(N,0),basis(N, 0),basis(N,0),basis(N,0),basis(N,0),basis(N,0),basis(N,0),basis(N,0),basis(N,0),basis(2, 1))  # start with an excited atom

# operators
a = tensor(destroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N), qeye(2))
b = tensor(qeye(N), destroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N), qeye(2))
c = tensor(qeye(N),qeye(N),destroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(2))
d = tensor(qeye(N),qeye(N),qeye(N),destroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(2))
e = tensor(qeye(N),qeye(N),qeye(N),qeye(N),destroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(2))
f = tensor(qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),destroy(N),qeye(N),qeye(N),qeye(N),qeye(2))
f1 = tensor(qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),destroy(N),qeye(N),qeye(N),qeye(2))
f2 = tensor(qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),destroy(N),qeye(N),qeye(2))
f3 = tensor(qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),destroy(N),qeye(2))
sm = tensor(qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N), destroy(2))

# Hamiltonian
H = wc1*a.dag()*a + wc2*b.dag()*b + wc3*c.dag()*c + wc4*d.dag()*d + wc5*e.dag()*e + wc6*f.dag()*f + wc7*f1.dag()*f1 + wc8*f2.dag()*f2 + wc9*f3.dag()*f3 + wa*sm.dag()*sm + g1*(a.dag()*sm + a*sm.dag()) + g2*(
            b.dag()*sm +b*sm.dag()) + g3*(c.dag()*sm + c*sm.dag()) + g4*(d.dag()*sm + d*sm.dag()) + g5*(e.dag()*sm + e*sm.dag()) + g6*(f.dag()*sm + f*sm.dag()) + g7*(f1.dag()*sm + f1*sm.dag()) + g8*(f2.dag()*sm + f2*sm.dag()) + g9*(f3.dag()*sm + f3*sm.dag())# + 20 * a.dag() * a * a.dag() * a

# The last term is the Kerr term

c_op_list = []

rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a)

rate = kappa * n_th_a
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a.dag())

rate = gamma
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm)

output = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, b.dag() * b, c.dag()*c,d.dag()*d,e.dag()*e,f.dag()*f,f1.dag()*f1,f2.dag()*f2,f3.dag()*f3, sm.dag() * sm])

fig, ax = plt.subplots(figsize=(8, 5))
#ax.plot(tlist, output.expect[0], label="$<n_1>$")
#ax.plot(tlist, output.expect[1], label="$<n_2>$")
#ax.plot(tlist, output.expect[2], label="$<n_3>$")
#ax.plot(tlist, output.expect[3], label="$<n_4>$")
#ax.plot(tlist, output.expect[4], label="$<n_5>$")
#ax.plot(tlist, output.expect[5], label="$<n_6>$")
#ax.plot(tlist, output.expect[6], label="$<n_7>$")
#ax.plot(tlist, output.expect[7], label="$<n_8>$")
#ax.plot(tlist, output.expect[8], label="$<n_9>$")
#ax.plot(tlist, output.expect[9], label="Atom excited state")
#ax.legend()
#ax.set_xlabel('Time (arbitrary units)')
#ax.set_ylabel('Occupation probability')
#ax.set_title('Rabi oscillations | Nine-cavity + one transmon | ');

# filename = r'C:\Users\dogak\Documents\Research\3D Algorithms\Multicavity\Notes\fig4.png'
# fig.savefig(filename)
print("hola")
list1=[max(output.expect[0]),max(output.expect[1]),max(output.expect[2]),max(output.expect[3]),max(output.expect[4]),max(output.expect[5]),max(output.expect[6]),max(output.expect[7]),max(output.expect[8])]
labels=["n1","n2","n3","n4","n5","n6","n7","n8","n9"]
ax.plot(labels,list1)
ax.set_xlabel('Mode #')
ax.set_ylabel('Maximum Occupation probability')
ax.set_title('9-Cavity + One Transmon Maximum Occupation Probability')
plt.show()