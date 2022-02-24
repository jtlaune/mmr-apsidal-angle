import numpy as np

mutots = np.logspace(-4,-8, 5) # mass of perturber
N = len(mutots)
dirname = [f"mup{mutot:0.2e}" for mutot in mutots]

Nsims = 5

h = np.ones(Nsims) * 0.03
q = np.ones(Nsims) * 1.
a0 = np.ones(Nsims) * 1.0
alpha2_0 = np.ones(Nsims) * 1.2
Te = np.ones(Nsims) * 1e3
Tm = np.ones(Nsims) * 1e5
T = np.ones(Nsims) * 10000
e10 = np.ones(Nsims) * 0.0
e20 = np.ones(Nsims) * 0.0
g1_0 = np.ones(Nsims) * 0.
g2_0 = np.ones(Nsims) * 0.
N_tps = np.ones(Nsims) * 5
Nout = np.ones(Nsims) * 1000
da_tps = np.ones(Nsims) * 0.1 # 1/2 full width
name = [f"ep{e10[i]:0.3f}" for i in range(Nsims)]

RUN_PARAMS = np.column_stack(
    (
        h,
        q,
        mutots,
        a0,
        alpha2_0,
        T,
        Te,
        Tm,
        e10,
        e20,
        name,
        dirname,
        g1_0,
        g2_0,
        N_tps,
        Nout,
        da_tps
    )
)
