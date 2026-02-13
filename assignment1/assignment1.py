import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D



df = pd.read_csv('/Users/aakash/Downloads/Kinetics.csv')
s1 = df['S1'].values
s2 = df['S2'].values
rate = df['Rate'].values

# Defining mechanism functions
def sequential_mech(X, Vmax, Km1, Km2, Ks1):
    S1, S2 = X
    return (Vmax * S1 * S2) / (Ks1 * Km2 + Km2 * S1 + Km1 * S2 + S1 * S2)

def ping_pong_mech(X, Vmax, Km1, Km2):
    S1, S2 = X
    return (Vmax * S1 * S2) / (Km2 * S1 + Km1 * S2 + S1 * S2)

# ques 1.1
# initial guesses: Vmax=1.0, Km1=0.1, Km2=0.1, Ks1=0.1
try:
    popt_seq, _ = curve_fit(sequential_mech, (s1, s2), rate, p0=[1, 0.1, 0.1, 0.1])
    pred_seq = sequential_mech((s1, s2), *popt_seq)
    r2_seq = r2_score(rate, pred_seq)
    
    popt_pp, _ = curve_fit(ping_pong_mech, (s1, s2), rate, p0=[1, 0.1, 0.1])
    pred_pp = ping_pong_mech((s1, s2), *popt_pp)
    r2_pp = r2_score(rate, pred_pp)

    print(f"R-squared Sequential: {r2_seq:.5f}")
    print(f"R-squared Ping-Pong: {r2_pp:.5f}")
    
    best_popt = popt_seq if r2_seq > r2_pp else popt_pp
    mechanism = "Sequential" if r2_seq > r2_pp else "Ping-Pong"
    print(f"\nIdentified Mechanism: {mechanism}")

except Exception as e:
    print(f"Error in fitting: {e}")

# ques 1.2
plt.figure(figsize=(10, 6))
for s2_val in sorted(df['S2'].unique()):
    subset = df[df['S2'] == s2_val]
    plt.plot(1/subset['S1'], 1/subset['Rate'], 'o-', label=f'[S2] = {s2_val} mM')

plt.title('Lineweaver Burk Plot (Double Reciprocal)')
plt.xlabel('1/[S1] (1/mM)')
plt.ylabel('1/v (s/mM)')
plt.legend()
plt.grid(True)
plt.show()

Vmax_ext = best_popt[0]
Km1_ext = best_popt[1]
print(f"Extracted Vmax: {Vmax_ext:.4f} mM/s")
print(f"Extracted Km for S1: {Km1_ext:.4f} mM")

# ques 1.3
MW_S1 = 150 # g/mol 
S1_initial_gL = 100 # g/L 
S1_final_gL = 1 # g/L 

# convert g/L to mM (mmol/L): (g/L / g/mol) * 1000
s1_0 = (S1_initial_gL / MW_S1) * 1000
s1_t = (S1_final_gL / MW_S1) * 1000

time_sec = (1/Vmax_ext) * ((s1_0 - s1_t) + Km1_ext * np.log(s1_0/s1_t))

print(f"\nInitial S1: {s1_0:.2f} mM")
print(f"Final S1: {s1_t:.2f} mM")
print(f"Time required: {time_sec:.2f} seconds ({time_sec/60:.2f} minutes)")

#Additional plots 

Vmax, Km_S1, Km_S2 = Vmax_ext, Km1_ext, 0.1 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
s1_range = np.linspace(0.01, 1.0, 30)
s2_range = np.linspace(0.05, 1.0, 30)
S1_grid, S2_grid = np.meshgrid(s1_range, s2_range)
Rate_grid = (Vmax * S1_grid * S2_grid) / (Km_S2 * S1_grid + Km_S1 * S2_grid + S1_grid * S2_grid)

surf = ax.plot_surface(S1_grid, S2_grid, Rate_grid, cmap='viridis', alpha=0.6)
ax.scatter(s1, s2, rate, color='black', s=20, label='Experimental Data')
ax.set_title('Innovative Plot 2: 3D Global Rate Surface')
ax.set_xlabel('[S1] (mM)')
ax.set_ylabel('[S2] (mM)')
ax.set_zlabel('Rate (mM/s)')
plt.show()

pred_rate = (Vmax * s1 * s2) / (Km_S2 * s1 + Km_S1 * s2 + s1 * s2)
residuals = rate - pred_rate
plt.figure(figsize=(8, 4))
plt.scatter(pred_rate, residuals, color='purple', alpha=0.6)
plt.axhline(0, color='black', lw=1)
plt.title('Residual Distribution')
plt.xlabel('Predicted Rate (mM/s)')
plt.ylabel('Residuals (Observed - Predicted)')
plt.show()

S0_mM = (100 / 150) * 1000 
St_mM = np.linspace(S0_mM, (0.5/150)*1000, 100) # Simulating down to 0.5 g/L
t_sim = (1/Vmax) * ((S0_mM - St_mM) + Km_S1 * np.log(S0_mM / St_mM))

plt.figure(figsize=(8, 5))
plt.plot(t_sim, St_mM * 150 / 1000, color='teal', lw=2.5)
plt.axhline(1, color='orange', ls='--', label='Target (1 g/L)')
plt.fill_between(t_sim, St_mM * 150 / 1000, color='teal', alpha=0.1)
plt.title('Progress Curve')
plt.xlabel('Time (seconds)')
plt.ylabel('[S1] Concentration (g/L)')
plt.legend()
plt.show()
