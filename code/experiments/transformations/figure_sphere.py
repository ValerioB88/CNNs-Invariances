import framework_utils
import numpy as np
import matplotlib.pyplot as plt
def from_azi_incl_to_rad(azi, incl):
    return [np.sin(incl) * np.cos(azi), np.sin(azi) * np.sin(incl), np.cos(incl)]

plt.close('all')
_, ax = framework_utils.create_sphere()


##
# v = [np.deg2rad(-10), np.deg2rad(-10)]
# pv = np.rad2deg(from_azi_incl_to_rad(v[0], v[1]))

# azi = 20
# incl = 90 - x

for azi in np.linspace(0, 360, 10):
    for incl in np.linspace(30, 120, 10):
        v = from_azi_incl_to_rad(np.deg2rad(azi), np.deg2rad(incl))
         # framework_utils.add_norm_vector(v, ax=ax)
        ax.plot(v[0], v[1], v[2], 'go', alpha=0.8)
##
plt.axis('off')
save_path = './results/transformations/figures/single_figs/'

plt.savefig(save_path + 'sphere3.svg', format='svg')
