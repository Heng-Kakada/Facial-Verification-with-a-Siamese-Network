import argparse

# Initialize parser
parser = argparse.ArgumentParser(
    description='Just an example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--name', default='piko', type=str)
parser.add_argument('--width', default=10.5, type=int)

args =vars( parser.parse_args() )
print(args['name'])