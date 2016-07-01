#!/usr/bin/env python3

import argparse
import contextlib
from os import linesep
from re import match, sub
import sys


FORMATS=["PDB", "PUREMD", "REAXFF_FORTRAN"]


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def count_atoms(input_file, input_format):
    count = 0
    regex = lambda x: False
    if input_format == "PDB":
        regex = lambda l: match(r'ATOM  ', l) != None
    elif input_format == "PUREMD":
        regex = lambda l: match(r'BOXGEO', l) == None
    elif input_format == "REAXFF_FORTRAN":
        regex = lambda l: match(r'HETATM', l) != None

    with open(input_file, 'r') as infile:
        for line in infile:
            if regex(line):
                count = count + 1

    return count


def convert(args):
    patterns = { \
            ("PDB", "PUREMD"): [ \
                lambda l, **kwargs: (sub(r'^ATOM  ([ 0-9]{5}) (.{4}).{14}([. 0-9]{8})([. 0-9]{8})([. 0-9]{8}).{22}([ a-zA-z]{2}).{2}$',
                    r'\1 \2 \6 \3 \4 \5', l), True) if not match(r'ATOM  ', l) == None else (l, False),
                lambda l, **kwargs: (sub(r'^CRYST1([. 0-9]{9})([. 0-9]{9})([. 0-9]{9})([. 0-9]{7})([. 0-9]{7})([. 0-9]{7}) .{15}$',
                    r'BOXGEO \1 \2 \3 \4 \5 \6', l), False) if not match(r'CRYST1', l) == None else (l, False),
                lambda l, **kwargs: (l + str(kwargs['atom_count']) + linesep, True)
                    if not match(r'BOXGEO', l) == None else (l, False)
            ],
            ("PDB", "REAXFF_FORTRAN"): [ \
                lambda l, **kwargs: ('HETATM {:5d} {:<5} {:3} {:1} {:5}{:10.5f}{:10.5f}{:10.5f} {:5}{:3d}{:2d} {:8.5f}'.format(
                        int(l[6:11].strip()), l[76:78].strip(), '  ', ' ', '     ', float(l[30:38].strip()),
                        float(l[38:46].strip()), float(l[46:54].strip()), l[76:78].strip(),
                        1, 1, 0.0) + linesep, True)
                    if not match(r'ATOM  ', l) == None else (l, False),
                lambda l, **kwargs: (l, True),
                                    ],
            ("PUREMD", "PDB"): [ \
                #TODO
                    ],
            ("PUREMD", "REAXFF_FORTRAN"): [ \
                #TODO
                    ],
            ("REAXFF_FORTRAN", "PDB"): [ \
                #TODO
                    ],
            ("REAXFF_FORTRAN", "PUREMD"): [ \
                #TODO
                    ],
            }

    ac = count_atoms(args.input_file, args.input_format)

    with open(args.input_file, 'r') as infile, smart_open(args.output_file) as outfile:
        for line in infile:
            for p in patterns[(args.input_format, args.output_format)]:
                try:
                    (line, matched) = p(line, atom_count=ac)
                except Exception as exc:
                   raise RuntimeError('Invalid input format')
                if matched:
                    break
            else:
                continue
            outfile.write(line)


def replicate(args):
    #TODO
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Munge geometry file(s) for molecular dynamics simulations.')
    subparsers = parser.add_subparsers(dest='cmd')
    subparsers.required = True

    # convert subcommand
    parser_convert = subparsers.add_parser('convert', aliases=['c', 'co'],
            help='Convert between geometry formats.')
    parser_convert.add_argument('input_format', metavar='in_format',
            choices=FORMATS, help='input format')
    parser_convert.add_argument('input_file', metavar='in_file',
            help='input geometry file')
    parser_convert.add_argument('output_format', metavar='out_format',
            choices=FORMATS, help='output format')
    parser_convert.add_argument('output_file', metavar='out_file', nargs='?',
            help='output geometry file')
    parser_convert.set_defaults(func=convert)

    # replicate subcommand
    parser_replicate = subparsers.add_parser('replicate', aliases=['r', 'rep'],
            help='Replicate geometry by prescribed factors in X, Y, and Z dimensions.')
    parser_replicate.add_argument('input_format', metavar='in_format',
            choices=FORMATS, help='input format')
    parser_replicate.add_argument('input_file', metavar='in_file',
            help='input geometry file')
    parser_replicate.add_argument('X repl', metavar='X', type=int,
            help='replication factor in X dimension')
    parser_replicate.add_argument('Y repl', metavar='Y', type=int,
            help='replication factor in Y dimension')
    parser_replicate.add_argument('Z repl', metavar='Z', type=int,
            help='replication factor in Z dimension')
    parser_replicate.add_argument('output_file', metavar='out_file', nargs='?',
            help='output geometry file')
    parser_replicate.set_defaults(func=replicate)

    # parse args and take action
    args = parser.parse_args()
    args.func(args)
