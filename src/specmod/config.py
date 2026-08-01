# Configuration File for SpecMod


SPECTRAL = dict(

    ### BINNING PARAMS
    ## min freq, max freq, # of bins
    BIN_PARS = {"smin": 0.001, "smax": 200, "bins": 151},
    ### SNR PARAMS
    ## minimum SNR required for pass
    SNR_TOLERENCE = 3,
    ## minimum number of points above SNR to pass
    MIN_POINTS = 10,

    ASSERT_BANDWIDTHS = False,
    ## bands to evaluate SNR (like Shearer)
    S_BANDS = [(2, 4), (4, 6), (6, 8)],


    ### SNR METHOD

    # if noise is shorter than signal - scale with np.sqrt(len(signal)/len(noise))
    SCALE_PARSEVAL = True,

    # BW_METHOD = 1,
    BW_METHOD = 2,

    ROTATE_NOISE = True,

    # ROT_METHOD = 1, # actual rotation, quite aggressive
    ROT_METHOD = 2, # non-linear boosting, bigger correction closer to freq lims

    # ROT_PARS = {'bcond': 0, 'fcond': -1, 'inc': 0.001}, # ROT_METHOD = 1
    ROT_PARS = {'inc': 0.05, 'space': [1e-3, 1+1e-3]}, # ROT_METHOD = 2


    ### VIS OPTS
    PLOT_COLUMNS = 3

)

MODELS = dict(

    MODEL = "BRUNE",
    MOTION = "velocity"

)

FITTING = dict(
    ### HOW MANY COLS TO PLOT
    PLOT_COLUMNS = 3
)
