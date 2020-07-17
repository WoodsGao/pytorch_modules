import argparse
import os
import os.path as osp
import sys

import hydra


def train(cfg):
    print(cfg.pretty())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Your config file path.')
    parser.add_argument('--strict',
                        action='store_true',
                        help='Strict mode for hydra.')
    opt, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    hydra_wrapper = hydra.main(config_path=osp.join(os.getcwd(),
                                                    opt.config_path),
                               strict=opt.strict)
    hydra_wrapper(train)()
