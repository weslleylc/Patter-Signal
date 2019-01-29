"""
    Signal module consists of all the basic algorithms and their implementation
"""


from .__about__ import (
    __package_name__,
    __title__,
    __author__,
    __author_email__,
    __license__,
    __copyright__,
    __version__,
    __version_info__,
    __url__,
)

from .feature_builder import (
     ECG,
     DefaultSignal,
     Gaussian,
     LeftInverseRayleigh,
     MathematicalModel,
     MexicanHat,
     PrintMexicanHat,
     Rayleigh,
     RightInverseRayleigh,
     TemplateEvaluetor,
 )


from .utils import (
     consume_process,
     consume_evaluetor,
     evaluete_models,
     mp_apply,
 )

__all__ = (
    # main classes
    "ECG",
    "DefaultSignal",
    "Gaussian",
    "LeftInverseRayleigh",
    "MathematicalModel",
    "MexicanHat",
    "PrintMexicanHat",
    "Rayleigh",
    "RightInverseRayleigh",
    "TemplateEvaluetor",
    # utils functions
    "consume_process",
    "consume_evaluetor",
    "evaluete_models",
    "mp_apply",
    # Metadata attributes
    "__package_name__",
    "__title__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "__version__",
    "__version_info__",
    "__url__",
)
