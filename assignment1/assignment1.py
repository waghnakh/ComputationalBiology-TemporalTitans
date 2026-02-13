import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chisquare



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
    chi2_seq, _ = chisquare(f_obs=rate, f_exp=pred_seq)
    
    popt_pp, _ = curve_fit(ping_pong_mech, (s1, s2), rate, p0=[1, 0.1, 0.1])
    pred_pp = ping_pong_mech((s1, s2), *popt_pp)
    r2_pp = r2_score(rate, pred_pp)
    chi2_pp, _ = chisquare(f_obs=rate, f_exp=pred_pp)

    print(f"R-squared Sequential: {r2_seq:.5f}")
    print(f"R-squared Ping-Pong: {r2_pp:.5f}")

    print(f"Chi-squared Sequential: {chi2_seq:.5e}")
    print(f"Chi-squared Ping-Pong : {chi2_pp:.5e}")
    
    best_popt = popt_seq if chi2_seq < chi2_pp else popt_pp
    mechanism = "Sequential" if chi2_seq < chi2_pp else "Ping-Pong"
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

#Question 2.4 (D) 

def plot_case(ax, v6_max, D, title):
    # X-as range (v1)
    v1 = np.linspace(0, 10, 100)
    
    # De relatie: v6 moet altijd v1 + D zijn
    v6_needed = v1 + D
    
    
    ax.plot(v1, v6_needed, label=f'Needed: $v_6 = v_1 + {D}$', color='blue', linewidth=2)
    
    
    ax.axhline(y=v6_max, color='red', linestyle='--', label=f'Ceiling: $v_{6,max} = {v6_max}$')
    
    
    if v6_max >= D:
        # Het snijpunt is waar v1 + D = v6_max  -> v1 = v6_max - D
        limit_v1 = v6_max - D
        
        valid_v1 = np.linspace(0, limit_v1, 100)
        valid_v6 = valid_v1 + D
        
        ax.fill_between(valid_v1, valid_v6, v6_max, color='green', alpha=0.3, label='Feasible Region')
        ax.scatter([limit_v1], [v6_max], color='black', zorder=5)
        ax.text(limit_v1, v6_max + 0.5, 'Max grens', ha='center')
    else:
        # Als het onmogelijk is
        ax.text(5, (v6_max + D)/2, 'IMPOSSIBLE\n(Starting point > Max value)', 
                ha='center', color='red', fontsize=12, fontweight='bold')

    # Opmaak
    ax.set_ylim(0, 15)
    ax.set_xlim(0, 10)
    ax.set_xlabel('$v_1$ (Flux Cyclus)')
    ax.set_ylabel('$v_6$ (Flux Enzym)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


plot_case(ax1, v6_max=12, D=4, title='Case 1: $v_{6,max} \geq D$ (Possible)')


plot_case(ax2, v6_max=3, D=5, title='Case 2: $v_{6,max} < D$ (Impossible)')

plt.tight_layout()
plt.show()

#Question 2.5 (E) 


plt.figure(figsize=(12, 8))


v1 = np.linspace(0, 10, 200)
D = 3.0
v6_line = v1 + D


plt.plot(v1, v6_line, color='black', linewidth=2, label='Mass Balance: $v_6 = v_1 + 3.0$')


cap_A = 12.0
limit_A = cap_A - D


cap_B = 8.0
limit_B = cap_B - D


cap_C = 5.0
limit_C = cap_C - D



# Feasible Space A 
v1_A = np.linspace(0, limit_A, 100)
plt.plot(v1_A, v1_A + D, color='blue', linewidth=10, alpha=0.3, label='Feasible Space A')
plt.axhline(y=cap_A, color='darkblue', linestyle='--', label='Cap A (Baseline): $V_{max}$ is high')

# Feasible Space B 
v1_B = np.linspace(0, limit_B, 100)
plt.plot(v1_B, v1_B + D, color='orange', linewidth=7, alpha=0.6, label='Feasible Space B')
plt.axhline(y=cap_B, color='orange', linestyle='--', label='Cap B (Lower $V_{max}$): Ceiling drops')

# Feasible Space C 
v1_C = np.linspace(0, limit_C, 100)
plt.plot(v1_C, v1_C + D, color='red', linewidth=4, alpha=1.0, label='Feasible Space C')
plt.axhline(y=cap_C, color='red', linestyle='--', label='Cap C (Higher $K_m$): Effective cap drops further')


plt.ylim(0, 15)
plt.xlim(0, 10)
plt.xlabel('Flux $v_1$ (Upper Cycle)', fontsize=12)
plt.ylabel('Flux $v_6$ (Lower Cycle)', fontsize=12)
plt.title('Impact of $V_{max}$ and $K_m$ on the $v_1$ vs $v_6$ Solution Space', fontsize=14)


plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10, framealpha=0.9)


plt.tight_layout()
plt.show()
