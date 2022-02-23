import numpy as np

mutots = [1e-6] # mass of perturber
N = len(mutots)
name = f"nbodyMigTrap"
dirname = [f"mup{mutot:0.2e}" for mutot in mutots]

h = 0.03
q = 1.
a0 = 1.0
alpha2_0 = 1.4
Te = 1e3
Tm = 1e5
T = 1.*Tm
e10 = 0.0
e20 = 0.0
g1_0 = 0.
g2_0 = 0.
N_tps = 25
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
    )
)
