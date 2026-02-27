import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D



states = ["Exon", "Intron"]
pi = np.array([0.5, 0.5]) # Table 2
A = np.array([            # Table 1
    [0.9, 0.1],  # Exon  (Exon, Intron)
    [0.2, 0.8]   # Intron  (Exon, Intron)
])
emit = {                  # Table 3
    "Exon":   {"A": 0.25, "U": 0.25, "G": 0.25, "C": 0.25},
    "Intron": {"A": 0.40, "U": 0.40, "G": 0.05, "C": 0.15},
}

def viterbi(obs):
    obs = list(obs)
    T = len(obs)
    N = len(states)

    logpi = np.log(pi)
    logA = np.log(A)

    logB = np.zeros((N, T))
    for j, st in enumerate(states):
        for t in range(T):
            logB[j, t] = np.log(emit[st][obs[t]])

    dp = np.full((N, T), -np.inf)
    ptr = np.zeros((N, T), dtype=int)
    dp[:, 0] = logpi + logB[:, 0]

    for t in range(1, T):
        for j in range(N):
            candidates = dp[:, t-1] + logA[:, j]
            ptr[j, t] = int(np.argmax(candidates))
            dp[j, t] = candidates[ptr[j, t]] + logB[j, t]

    best_last = int(np.argmax(dp[:, -1]))
    best_logprob = float(dp[best_last, -1])

    path_idx = [best_last]
    for t in range(T-1, 0, -1):
        path_idx.append(ptr[path_idx[-1], t])
    path_idx = path_idx[::-1]
    best_path = [states[i] for i in path_idx]

    scores = np.exp(dp - dp.max(axis=0, keepdims=True))
    scores = scores / scores.sum(axis=0, keepdims=True)

    return best_path, best_logprob, dp, scores

def viterbi_report(seq):
    path, logp, dp, scores = viterbi(seq)
    df = pd.DataFrame({
        "pos": np.arange(1, len(seq) + 1),
        "obs": list(seq),
        "Viterbi_state": path,
        "score_exon": scores[0, :],
        "score_intron": scores[1, :],
    })
    return df, logp

alpha_seq = "AGCGC"
beta_seq  = "AUUAU"

df_alpha, logp_alpha = viterbi_report(alpha_seq)
df_beta,  logp_beta  = viterbi_report(beta_seq)

print("Task 1 & 2: Viterbi decoding")
print("\nSequence alpha =", alpha_seq)
print(df_alpha.to_string(index=False))
print(f"Best-path probability (alpha) ≈ {np.exp(logp_alpha):.3e}")

print("\nSequence beta  =", beta_seq)
print(df_beta.to_string(index=False))
print(f"Best-path probability (beta)  ≈ {np.exp(logp_beta):.3e}")

def infer_mechanism(viterbi_states):
    exon_count = sum(s == "Exon" for s in viterbi_states)
    intr_count = len(viterbi_states) - exon_count
    return "I (Transcriptional Hijack)" if exon_count >= intr_count else "II (Splicing Sabotage)"

print("\nInferred mechanism for alpha:", infer_mechanism(df_alpha["Viterbi_state"].tolist()))
print("Inferred mechanism for beta :", infer_mechanism(df_beta["Viterbi_state"].tolist()))



p_ode = dict(
    mA=2.35, mB=2.35,   # Max transcription rates
    gammaA=1.0, gammaB=1.0, # mRNA degradation
    kPA=1.0, kPB=1.0,  # Translation rates
    thetaA=0.21, thetaB=0.21, # Binding thresholds
    nA=3, nB=3,  # Hill coefficients
    dPA=1.0, dPB=1.0  # Protein degradation
)

y0_ode = [0.8, 0.8, 0.8, 0.8] # [mA, mB, pA, pB] (from Table 5)

def ode_normal(t, y):
    mA, mB, pA, pB = y
    p = p_ode
    dmA = p["mA"] * (p["thetaA"]**p["nA"] / (p["thetaA"]**p["nA"] + pB**p["nB"])) - p["gammaA"] * mA
    dmB = p["mB"] * (p["thetaB"]**p["nB"] / (p["thetaB"]**p["nB"] + pA**p["nA"])) - p["gammaB"] * mB
    dpA = p["kPA"] * mA - p["dPA"] * pA
    dpB = p["kPB"] * mB - p["dPB"] * pB
    return [dmA, dmB, dpA, dpB]

def ode_patient_alpha(t, y):
    mA, mB, pA, pB = y
    p = p_ode
    dmA = p["mA"] * (p["thetaA"]**p["nA"] / (p["thetaA"]**p["nA"] + pB**p["nB"])) - p["gammaA"] * mA
    dmB = p["mB"] - p["gammaB"] * mB 
    dpA = p["kPA"] * mA - p["dPA"] * pA
    dpB = p["kPB"] * mB - p["dPB"] * pB
    return [dmA, dmB, dpA, dpB]

t_eval_ode = np.linspace(0, 30, 1000)
sol_norm = solve_ivp(ode_normal, [0, 30], y0_ode, t_eval=t_eval_ode)
sol_alpha = solve_ivp(ode_patient_alpha, [0, 30], y0_ode, t_eval=t_eval_ode)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(sol_norm.t, sol_norm.y[2], 'g-', lw=4, alpha=0.6, label="Protein A (Tumor Suppressor)")
axes[0].plot(sol_norm.t, sol_norm.y[3], 'r--', lw=2, label="Protein B (Oncogene)")
axes[0].set_title("Time Series: Normal State")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Concentration (M)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(sol_alpha.t, sol_alpha.y[2], 'g-', lw=2, label="Protein A (Tumor Suppressor)")
axes[1].plot(sol_alpha.t, sol_alpha.y[3], 'r-', lw=2, label="Protein B (Oncogene)")
axes[1].set_title("Time Series: Patient Alpha (Mech I)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Concentration (M)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(sol_norm.y[2], sol_norm.y[3], 'k--', lw=2, label="Normal Trajectory")
axes[2].plot(sol_alpha.y[2], sol_alpha.y[3], 'b-', lw=2, label="Patient Alpha Trajectory")
axes[2].set_title("Protein Phase Portrait (ODE)")
axes[2].set_xlabel("Protein A (Tumor Suppressor)")
axes[2].set_ylabel("Protein B (Oncogene)")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()




p_sde = dict(
    aA=1.0, aB=0.25,                 
    bA=0.0005, bB=0.0005,            
    cA=2.0, cB=0.5,                 
    mA=2.35, mB=2.35,     # Transcription
    betaA=2.35, betaB=2.35,  # Splicing
    gammaA=1.0, gammaB=1.0, # mRNA degradation
    kPA=1.0, kPB=1.0,   # Translation
    deltaPA=1.0, deltaPB=1.0, # Protein degradation
    thetaA=0.21, thetaB=0.21, # Thresholds
    nA=3, nB=3, # Hill coefficients
    sigma1A=0.05, sigma2A=0.05, # Specific Noise (Gene A)
    sigma1B=0.05, sigma2B=0.05  # Specific Noise (Gene B)
)

def simulate_sdevelo(is_patient_beta=False, T=30, dt=0.01, seed=42):
    np.random.seed(seed)
    steps = int(T/dt) + 1
    t = np.linspace(0, T, steps)
    
    uA, sA, pA = np.full(steps, 0.8), np.full(steps, 0.8), np.full(steps, 0.8)
    uB, sB, pB = np.full(steps, 0.8), np.full(steps, 0.8), np.full(steps, 0.8)
    
    p = p_sde
    sq_dt = np.sqrt(dt)

    for i in range(1, steps):
        dW1A = np.random.normal(0, sq_dt) * p["sigma1A"]
        dW2A = np.random.normal(0, sq_dt) * p["sigma2A"]
        dW1B = np.random.normal(0, sq_dt) * p["sigma1B"]
        dW2B = np.random.normal(0, sq_dt) * p["sigma2B"]
        
        transcription_A = p["mA"] * (p["thetaA"]**p["nA"] / (p["thetaA"]**p["nA"] + pB[i-1]**p["nB"]))
        uA[i] = uA[i-1] + (transcription_A - p["betaA"]*uA[i-1])*dt + dW1A
        sA[i] = sA[i-1] + (p["betaA"]*uA[i-1] - p["gammaA"]*sA[i-1])*dt + dW2A
        pA[i] = pA[i-1] + (p["kPA"]*sA[i-1] - p["deltaPA"]*pA[i-1])*dt
        
        if is_patient_beta:
            splicing_B_rate = p["betaB"] * (p["thetaB"]**p["nB"] / (p["thetaB"]**p["nB"] + pA[i-1]**p["nA"]))
            uB[i] = uB[i-1] + (p["mB"] - splicing_B_rate*uB[i-1])*dt + dW1B
            sB[i] = sB[i-1] + (splicing_B_rate*uB[i-1] - p["gammaB"]*sB[i-1])*dt + dW2B
        else:
            transcription_B = p["mB"] * (p["thetaB"]**p["nB"] / (p["thetaB"]**p["nB"] + pA[i-1]**p["nA"]))
            uB[i] = uB[i-1] + (transcription_B - p["betaB"]*uB[i-1])*dt + dW1B
            sB[i] = sB[i-1] + (p["betaB"]*uB[i-1] - p["gammaB"]*sB[i-1])*dt + dW2B
            
        pB[i] = pB[i-1] + (p["kPB"]*sB[i-1] - p["deltaPB"]*pB[i-1])*dt
        
        uA[i], sA[i], pA[i] = max(uA[i], 0), max(sA[i], 0), max(pA[i], 0)
        uB[i], sB[i], pB[i] = max(uB[i], 0), max(sB[i], 0), max(pB[i], 0)

    return t, np.column_stack((uA, uB, sA, sB, pA, pB))

tN, xN = simulate_sdevelo(is_patient_beta=False)
tB, xB = simulate_sdevelo(is_patient_beta=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(tB, xB[:, 1], 'r-', lw=2, label="Unspliced pre-mRNA B (uB)")
axes[0].plot(tB, xB[:, 3], 'b-', lw=2, label="Mature spliced mRNA B (sB)")
axes[0].set_title("RNA Dynamics: Splicing Sabotage")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Concentration (M)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(tB, xB[:, 4], 'g-', lw=2, label="Protein A (Tumor Suppressor)")
axes[1].plot(tB, xB[:, 5], 'r-', lw=2, label="Protein B (Oncogene)")
axes[1].set_title("Protein Time Series: Patient Beta")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Concentration (M)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(xN[:, 4], xN[:, 5], 'k--', lw=2, label="Normal Trajectory")
axes[2].plot(xB[:, 4], xB[:, 5], 'r-', lw=2, label="Patient Beta Trajectory")
axes[2].set_title("Protein Phase Portrait (SDEVelo)")
axes[2].set_xlabel("Protein A (Tumor Suppressor)")
axes[2].set_ylabel("Protein B (Oncogene)")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Task 3: Comparative Diagnosis - Phase Portraits', fontsize=16)

axes[0].plot(sol_norm.y[2], sol_norm.y[3], 'k--', lw=2, alpha=0.7, label="Normal State")
axes[0].plot(sol_alpha.y[2], sol_alpha.y[3], 'b-', lw=2, label="Patient Alpha Trajectory")
axes[0].set_title("Early-Stage: Transcriptional Hijack (ODE)")
axes[0].set_xlabel("Protein A (Tumor Suppressor)")
axes[0].set_ylabel("Protein B (Oncogene)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(xN[:, 4], xN[:, 5], 'k--', lw=2, alpha=0.7, label="Normal State")
axes[1].plot(xB[:, 4], xB[:, 5], 'r-', lw=2, label="Patient Beta Trajectory")
axes[1].set_title("Aggressive: Splicing Sabotage (SDEVelo)")
axes[1].set_xlabel("Protein A (Tumor Suppressor)")
axes[1].set_ylabel("Protein B (Oncogene)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

#Bonus question
alpha = 2.0
beta  = 1.1
gamma = 1.0
delta = 0.9

R_star = gamma / delta
E_star = alpha / beta

J = np.array([[0, -beta * R_star], 
              [delta * E_star, 0]])
eigenvalues = np.linalg.eigvals(J)

print("\nBonus: Stability Analysis")
print(f"Non-trivial Equilibrium (R*, E*) = ({R_star:.4f}, {E_star:.4f})")
print(f"Eigenvalues of Jacobian at Equilibrium: {eigenvalues}")
print("Because the eigenvalues are purely imaginary, the system is a neutrally stable center.\n")

def metabolic_rhs(t, z):
    R, E = z
    dR = alpha*R - beta*R*E
    dE = -gamma*E + delta*R*E
    return [dR, dE]

sol = solve_ivp(metabolic_rhs, [0, 20], [1.0, 0.5], t_eval=np.linspace(0, 20, 1000))

R_vals = np.linspace(0, 3.5, 40)
E_vals = np.linspace(0, 3.5, 40)
RR, EE = np.meshgrid(R_vals, E_vals)

dR = alpha*RR - beta*RR*EE
dE = -gamma*EE + delta*RR*EE

plt.figure(figsize=(8, 6))
plt.streamplot(R_vals, E_vals, dR, dE, density=1.2, color='lightgray')
plt.axhline(E_star, color='blue', linestyle="--", label="dR/dt=0 Nullcline")
plt.axvline(R_star, color='red', linestyle="--", label="dE/dt=0 Nullcline")
plt.plot(R_star, E_star, 'ko', markersize=8, label="Equilibrium (Center)")
plt.plot(sol.y[0], sol.y[1], 'g-', lw=2, label="Trajectory from R(0)=1, E(0)=0.5")
plt.plot(1.0, 0.5, 'go', markersize=6)

plt.title("Downstream Metabolic Effects: Resource vs Enzyme")
plt.xlabel("Cellular Resource (R)")
plt.ylabel("Growth-Promoting Enzyme (E)")
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.show()

#Additional supporting graphs/plots!!

def run_3d_sdevelo(T=30, dt=0.01):
    steps = int(T/dt) + 1
    uA, sA, pA = np.full(steps, 0.8), np.full(steps, 0.8), np.full(steps, 0.8)
    uB, sB, pB = np.full(steps, 0.8), np.full(steps, 0.8), np.full(steps, 0.8)
    sq_dt = np.sqrt(dt)
    p = p_sde

    for i in range(1, steps):
        dW1A = np.random.normal(0, sq_dt) * p["sigma1A"]
        dW2A = np.random.normal(0, sq_dt) * p["sigma2A"]
        dW1B = np.random.normal(0, sq_dt) * p["sigma1B"]
        dW2B = np.random.normal(0, sq_dt) * p["sigma2B"]
        
        trans_A = p["mA"] * (p["thetaA"]**p["nA"] / (p["thetaA"]**p["nA"] + pB[i-1]**p["nB"]))
        uA[i] = uA[i-1] + (trans_A - p["betaA"]*uA[i-1])*dt + dW1A
        sA[i] = sA[i-1] + (p["betaA"]*uA[i-1] - p["gammaA"]*sA[i-1])*dt + dW2A
        pA[i] = pA[i-1] + (p["kPA"]*sA[i-1] - p["deltaPA"]*pA[i-1])*dt
        
        splice_B = p["betaB"] * (p["thetaB"]**p["nB"] / (p["thetaB"]**p["nB"] + pA[i-1]**p["nA"]))
        uB[i] = uB[i-1] + (p["mB"] - splice_B*uB[i-1])*dt + dW1B
        sB[i] = sB[i-1] + (splice_B*uB[i-1] - p["gammaB"]*sB[i-1])*dt + dW2B
        pB[i] = pB[i-1] + (p["kPB"]*sB[i-1] - p["deltaPB"]*pB[i-1])*dt
        
        uB[i], sB[i], pB[i], pA[i] = max(uB[i], 0), max(sB[i], 0), max(pB[i], 0), max(pA[i], 0)
        
    return pA, pB, uB

pA_3d, pB_3d, uB_3d = run_3d_sdevelo(T=40)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pA_3d, pB_3d, uB_3d, c=np.linspace(0, 40, len(pA_3d)), cmap='coolwarm', s=2)
ax.set_title("3D SDEVelo Manifold: The 'Ticking Time Bomb'", fontsize=14)
ax.set_xlabel('Protein A (Suppressor)', fontweight='bold')
ax.set_ylabel('Protein B (Oncogene)', fontweight='bold')
ax.set_zlabel('Pre-mRNA B (The Hidden Reservoir)', fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Time Elapsed (s)')
plt.show()

def ode_bifurcation(t, y, hijack_strength):
    mA, mB, pA, pB = y
    p = p_sde
    dmA = p["mA"] * (p["thetaA"]**p["nA"] / (p["thetaA"]**p["nA"] + pB**p["nB"])) - p["gammaA"] * mA
    repression_term = (p["thetaB"]**p["nB"] / (p["thetaB"]**p["nB"] + pA**p["nA"]))
    actual_transcription = p["mB"] * ((1 - hijack_strength)*repression_term + hijack_strength*1.0)
    dmB = actual_transcription - p["gammaB"] * mB 
    dpA = p["kPA"] * mA - p["deltaPA"] * pA
    dpB = p["kPB"] * mB - p["deltaPB"] * pB
    return [dmA, dmB, dpA, dpB]


ensemble_uB = []
time_array = np.linspace(0, 30, int(30/0.01) + 1)

for _ in range(30):
    _, _, uB_traj = run_3d_sdevelo(T=30)
    plt.plot(time_array, uB_traj, color='red', alpha=0.15, lw=1.5)
    ensemble_uB.append(uB_traj)

mean_uB = np.mean(ensemble_uB, axis=0)
plt.plot(time_array, mean_uB, color='darkred', lw=3, label="Ensemble Average (Deterministic Trend)")

plt.title("Stochastic Ensemble: Noise-Driven Splicing Sabotage", fontsize=14)
plt.xlabel("Time (s)")
plt.ylabel("Pre-mRNA B Concentration (M)")
plt.text(5, 4.5, "Each faint red line is a unique cell simulation.\nNotice the variance in peak height due to molecular noise.", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


pA_grid = np.linspace(0.1, 3.5, 60)
pB_grid = np.linspace(0.1, 3.5, 60)
PA, PB = np.meshgrid(pA_grid, pB_grid)

dpA_landscape = p_ode["kPA"] * (p_ode["mA"] * (p_ode["thetaA"]**p_ode["nA"] / (p_ode["thetaA"]**p_ode["nA"] + PB**p_ode["nB"])) / p_ode["gammaA"]) - p_ode["dPA"] * PA
dmB_ss = (p_ode["mB"]) / p_ode["gammaB"] 
dpB_landscape = p_ode["kPB"] * dmB_ss - p_ode["dPB"] * PB

speed = np.sqrt(dpA_landscape**2 + dpB_landscape**2)
energy = np.log(speed + 0.05)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(PA, PB, energy, cmap='viridis_r', edgecolor='none', alpha=0.9)

ax.set_title("The 'Attractor Basin' of the Cancer State", fontsize=14)
ax.set_xlabel('Protein A (Tumor Suppressor)', fontweight='bold')
ax.set_ylabel('Protein B (Oncogene)', fontweight='bold')
ax.set_zlabel('Pseudo Potential Energy (U)', fontweight='bold')

ax.scatter([sol_alpha.y[2][-1]], [sol_alpha.y[3][-1]], [np.min(energy)], 
           color='red', s=100, label='Tumor Steady State', zorder=5)

plt.legend()
plt.show()


grid_size = 20
mA_vals = np.linspace(0.5, 5.0, grid_size)
mB_vals = np.linspace(0.5, 5.0, grid_size)
steady_state_pB = np.zeros((grid_size, grid_size))

for i, test_mA in enumerate(mA_vals):
    for j, test_mB in enumerate(mB_vals):
        temp_p_ode = p_ode.copy()
        temp_p_ode["mA"] = test_mA
        temp_p_ode["mB"] = test_mB
        
        def sweep_ode(t, y):
            mA, mB, pA, pB = y
            dmA = temp_p_ode["mA"] * (temp_p_ode["thetaA"]**temp_p_ode["nA"] / (temp_p_ode["thetaA"]**temp_p_ode["nA"] + pB**temp_p_ode["nB"])) - temp_p_ode["gammaA"] * mA
            dmB = temp_p_ode["mB"] * (temp_p_ode["thetaB"]**temp_p_ode["nB"] / (temp_p_ode["thetaB"]**temp_p_ode["nB"] + pA**temp_p_ode["nA"])) - temp_p_ode["gammaB"] * mB
            dpA = temp_p_ode["kPA"] * mA - temp_p_ode["dPA"] * pA
            dpB = temp_p_ode["kPB"] * mB - temp_p_ode["dPB"] * pB
            return [dmA, dmB, dpA, dpB]
        
        sol = solve_ivp(sweep_ode, [0, 50], [0.8, 0.8, 0.8, 0.8], t_eval=[50])
        steady_state_pB[j, i] = sol.y[3][-1] # Note: j is row (y axis mB), i is col (x axis mA)

plt.figure(figsize=(9, 7))
im = plt.imshow(steady_state_pB, extent=[min(mA_vals), max(mA_vals), min(mB_vals), max(mB_vals)], 
           origin='lower', cmap='magma', aspect='auto')

plt.contour(mA_vals, mB_vals, steady_state_pB, levels=[1.5], colors='white', linestyles='dashed', linewidths=2)

plt.colorbar(im, label='Steady State Protein B (Oncogene)')
plt.title("Cancer Phase Boundary: System Robustness\n(Varying Transcription Rates $m_A$ and $m_B$)", fontsize=14)
plt.xlabel("Tumor Suppressor Max Transcription Rate ($m_A$)")
plt.ylabel("Oncogene Max Transcription Rate ($m_B$)")
plt.text(4.0, 1.0, "Healthy\nRegime", color='white', fontweight='bold', ha='center')
plt.text(1.0, 4.0, "Cancer\nRegime", color='white', fontweight='bold', ha='center')
plt.show()


def logistic_map(r, x):
    return r * x * (1 - x)

n_points = 10000
r_values = np.linspace(2.5, 4.0, n_points)
iterations = 1000
last = 100

x = 1e-5 * np.ones(n_points)
r_plot = []
x_plot = []

for i in range(iterations):
    x = logistic_map(r_values, x)
    if i >= (iterations - last):
        r_plot.append(r_values)
        x_plot.append(x)

plt.figure(figsize=(12, 7))
plt.plot(r_plot, x_plot, ',k', alpha=0.25) 
plt.title("Generational Tumor Growth: Period Doubling Route to Chaos", fontsize=15)
plt.xlabel("Proliferation Drive 'r' (Driven by Oncogene Protein B Expression)", fontsize=12)
plt.ylabel("Tumor Population Density (Normalized)", fontsize=12)

plt.axvline(x=3.0, color='blue', linestyle='--', alpha=0.6)
plt.text(2.6, 0.1, "Stable Tumor\n(Early Stage)", color='blue', fontweight='bold')

plt.axvline(x=3.54, color='orange', linestyle='--', alpha=0.6)
plt.text(3.1, 0.1, "Boom Bust Cycles\n(Bifurcations)", color='orange', fontweight='bold')

plt.text(3.7, 0.1, "Deterministic Chaos\n(Aggressive Malignancy)", color='red', fontweight='bold')

plt.xlim(2.5, 4.0)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


r_chaos = 3.9  
x_cobweb = np.linspace(0, 1, 500)
y_cobweb = logistic_map(r_chaos, x_cobweb)

plt.figure(figsize=(8, 8))
plt.plot(x_cobweb, y_cobweb, 'r-', lw=2, label=f"Tumor Growth Map (r={r_chaos})")
plt.plot(x_cobweb, x_cobweb, 'k--', label="Steady State Line ($P_{n+1} = P_n$)")

x_current = 0.1
for _ in range(50):
    y_current = logistic_map(r_chaos, x_current)
    plt.plot([x_current, x_current], [x_current, y_current], 'b-', alpha=0.5)
    plt.plot([x_current, y_current], [y_current, y_current], 'b-', alpha=0.5)
    x_current = y_current

plt.title("Chaotic Tumor Attractor (Cobweb Plot)\nPatient Beta at Maximum Oncogene Drive", fontsize=14)
plt.xlabel("Tumor Population at Generation $n$", fontsize=12)
plt.ylabel("Tumor Population at Generation $n+1$", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
