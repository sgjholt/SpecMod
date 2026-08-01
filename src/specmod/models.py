# MODELS contains a set of functions for minimisation to seismic spectra.
# It can be modified as appropriate.
import numpy as np
from . import config as cfg

MODS = ["BRUNE", "BOATWRIGHT"]

# UTIL FUNCS
def which_model(mod):
    if mod in MODS:
        if mod == "BRUNE":
            return BRUNE_MODEL
        if mod == "BOATWRIGHT":
            return BOATWRIGHT_MODEL
    else:
        raise ValueError(f"Model {mod} not available. Choose from {MODS}.")


def scale_to_motion(motion, f):
    if motion.lower() == 'displacement':
        return 0

    elif motion.lower() == 'velocity':
        return np.log10(2*np.pi*f)

    elif motion.lower() == 'acceleration':
        return np.log10(np.power(2*np.pi*f,2))
    else:
        return None

# DEFAULT PARAMS FOR SOURCE MODE:
BRUNE_MODEL = (1, 2) # omega squared
BOATWRIGHT_MODEL = (2, 2) # omega cubed
#
MODEL = which_model(cfg.MODELS["MODEL"])
MOTION = cfg.MODELS["MOTION"]



# MINIMISATION FUNCTIONS
## Source model
def source(f, llpsp, fc):
    gam, n = MODEL
    loga = llpsp - (1/gam)*np.log10((1+(f/fc)**(gam*n)))
    return loga

# freq independent t-star attenuation model
def t_star(f, ts):
    return -(np.pi*f*ts / np.log(10))

# freq dependent t-star attenuation
def t_star_freq(f, ts, a):
    return -(np.pi*(f**(1-a))*ts / np.log(10))

# combine models
def simple_model(f, llpsp, fc, ts):
    global MOTION
    """
    Simple attenuated source model to minimise.
    """
    return source(f, llpsp, fc) + t_star(f, ts) + scale_to_motion(MOTION, f)

def simple_model_fdep(f, llpsp, fc, ts, a):
    """
    Simple model but with frequency dependent attenuation.
    """
    return source(f, llpsp, fc) + t_star_freq(f, ts, a) + scale_to_motion(MOTION, f)
