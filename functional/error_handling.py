from typing import List


def check_sal_dim_support(sal_dim: str, supp_modes: List = None):
    if supp_modes is None:
        supp_modes = ["spat", "temp", "spatiotemp"]
    if sal_dim not in supp_modes:
        raise ValueError("Saliency dimension '{}' is not supported! "
                         "Supported saliency dimensions are: {}".format(sal_dim, supp_modes))


def check_sal_type_support(sal_type: str, supp_modes: List = None):
    if supp_modes is None:
        supp_modes = ["att", "conf", "conf_att"]
    if sal_type not in supp_modes:
        raise ValueError("Saliency of type '{}' is not supported! "
                         "Supported saliency types are: {}".format(sal_type, supp_modes))
