from typing import List


def check_sal_type_support(sal_type: str, supp_modes: List = None):
    if supp_modes is None:
        supp_modes = ["spat", "temp", "spatiotemp"]
    if sal_type not in supp_modes:
        raise ValueError("Saliency of type '{}' is not supported! "
                         "Supported saliency types are: {}".format(sal_type, supp_modes))
