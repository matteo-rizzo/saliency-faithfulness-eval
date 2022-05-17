import argparse

from src.classes.tasks.ccc.multiframe.modules.saliency_tccnet.core.ModelSaliencyTCCNet import ModelSaliencyTCCNet
from src.functional.metrics import num_params


def main(ns: argparse.Namespace):
    hidden_size, kernel_size = ns.hidden_size, ns.kernel_size
    sal_types, sal_dims = ["att", "conf", "conf_att"], ["spat", "temp", "spatiotemp"]

    for sal_type in sal_types:
        for sal_dim in sal_dims:
            model = ModelSaliencyTCCNet(sal_type, sal_dim, hidden_size, kernel_size)

            print("\n------------------------------------------------------------------------------------------")
            print("\t\t Printing capacity for '{}' - '{}'".format(sal_type, sal_dim))
            print("------------------------------------------------------------------------------------------\n")

            print(num_params(model))
            model.print_network()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    namespace = parser.parse_args()
    main(namespace)
