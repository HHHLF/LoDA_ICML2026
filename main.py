import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='exps/dlora/imgr10.json',
                        help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/imga10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/cub10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/cifar10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/domainnet.json',
    #                     help='Json file of settings.')

    # parser.add_argument('--device', type=str, default='2')

    return parser


if __name__ == '__main__':
    main()
