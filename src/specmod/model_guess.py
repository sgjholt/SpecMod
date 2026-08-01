# CustomGuess is a file that contains functions that are specific to create
# a dictionary of pre-fit parameter guesses. Each guess function should be
# specific to a minimisation function in the Models.py file. Either of these
# files should be modified for a particular pair of minimisation / guess functions.
## imports
import numpy as np


def create_simple_guess(spectra):

    """
    This function creates the initial model parameter guesses for the
    function simple_model in Models.py.

    Each dictionary entry in guess should be :
        'trace id' : {'param1':param1_val, ..., 'paramN':paramN_val}

    The trace id must be the same as it appears in the spectra object.
    """

    guess = {}
    for ID, spec in spectra.group.items():
        try:
            inds = np.where((spec.signal.freq>=spec.signal.ubfreqs[0]) & (
                 spec.signal.freq<=spec.signal.ubfreqs[1]))
            # print(ID, inds)
            llpsp = np.log10(spec.signal.amp[inds].max())
            fc = spec.signal.freq[inds][spec.signal.amp[inds].argmax()]
            guess.update({ ID : {'llpsp':llpsp, 'fc': fc, 'ts': 0.01}})

        except IndexError:
            guess.update({ID: {'llpsp':None, 'fc': None, 'ts': None}})

    return guess

def create_simple_guess_fdep(spectra):
    guess = {}
    for ID, spec in spectra.group.items():
        try:
            inds = np.where((spec.signal.freq>=spec.signal.ubfreqs[0]) & (
                 spec.signal.freq<=spec.signal.ubfreqs[1]))
            # print(ID, inds)
            llpsp = np.log10(spec.signal.amp[inds].max())
            fc = spec.signal.freq[inds][spec.signal.amp[inds].argmax()]
            guess.update({ ID : {'llpsp':llpsp, 'fc': fc, 'ts': 0.01,
                                 'a':0.00001}})

        except IndexError:
            guess.update({ID: {'llpsp':None, 'fc': None, 'ts': None, 'a': None}})

    return guess
