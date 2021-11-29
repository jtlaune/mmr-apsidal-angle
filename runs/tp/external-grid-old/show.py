import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


j = 2
eps = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
eeqs = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
ratios = eeqs**2*2*j
for ep in eps:
    for ratio in ratios:
        e_eq = np.sqrt(ratio/(2*j))
        label = "{:0.2e}-{:0.2e}".format(ep, e_eq)
        figname = "external-grid-"+label+".png"
        image = Image.open(figname)
        image.show()
        print("ep=", ep)
        print("e_eq=", e_eq)
        input("Press Enter to continue...")
        image.close()
