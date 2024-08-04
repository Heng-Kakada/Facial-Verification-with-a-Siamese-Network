import argparse

parser = argparse.ArgumentParser(
    description='Samrt Home V2',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--camera', default=0, type=int, help='This Open Your Own Camera That Available On Your Computer Input')

args = vars(parser.parse_args())