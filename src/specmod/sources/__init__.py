"""Source, attenuation and motion models, and the registry that names them.

The spectrum being fitted is a product of three things, so this package keeps
them separate and composes them:

.. code-block:: text

    log10 A(f) = log10 S(f)  +  log10 D(f)  +  log10 G(f)
                 source         attenuation    motion

Splitting them matters because they are chosen independently and confused
easily. A Brune source with frequency-dependent attenuation is a legitimate
combination; so is a Boatwright source with constant ``t*``. The legacy code
bound the pair together at import time through ``MODEL = which_model(...)``,
which is why a Brune and a Boatwright could not be fitted in one session.

What a source model has to carry
--------------------------------
Not just a spectral shape. **The relation between corner frequency and source
radius belongs to the model**, and forgetting that is a specific trap:

Madariaga is omega-squared like Brune and sits at the *same* ``(gamma, n)``, so
adding it as a shape alone changes no fitted parameter — the fit is identical
— while the source radius it implies is quite different. Stress drop goes as
``r**-3``, so that is roughly an order of magnitude on identical data, arriving
silently. :attr:`SourceModel.corner_frequency_coefficient` exists so the
difference cannot be lost that way, and so that a model which changes nothing
about the fit still changes what is derived from it.
"""

from __future__ import annotations

from .attenuation import (
    ATTENUATION_MODELS,
    AttenuationModel,
    ConstantQ,
    FrequencyDependentQ,
    get_attenuation_model,
)
from .composite import SpectralModel, build_model, from_config
from .motion import motion_scaling
from .source import (
    SOURCE_MODELS,
    BoatwrightSource,
    BruneSource,
    SourceModel,
    get_source_model,
)

__all__ = [
    "ATTENUATION_MODELS",
    "SOURCE_MODELS",
    "AttenuationModel",
    "BoatwrightSource",
    "BruneSource",
    "ConstantQ",
    "FrequencyDependentQ",
    "SourceModel",
    "SpectralModel",
    "build_model",
    "from_config",
    "get_attenuation_model",
    "get_source_model",
    "motion_scaling",
]
