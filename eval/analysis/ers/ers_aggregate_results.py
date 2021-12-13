import argparse
import os

from auxiliary.settings import RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace


def main(ns: argparse.Namespace):
    sal_type, sal_dim, test_type = ns.sal_type, ns.sal_dim, ns.test_type
    path_to_results = os.path.join(test_type, sal_dim, sal_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--test_type", type=str, default="single")
    parser.add_argument("--sal_dim", type=str, default="spatiotemp")
    parser.add_argument("--sal_type", type=str, default="att")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
