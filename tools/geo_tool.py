#!/usr/bin/env python3

def convert(args):
    import contextlib
    from re import match, sub
    from os import linesep

    @contextlib.contextmanager
    def smart_open(filename=None):
        import sys

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
        from re import match

        count = 0
        regex = lambda x: False
        if input_format == "BGF":
            regex = lambda l: match(r'HETATM', l) != None
        elif input_format == "PDB":
            regex = lambda l: match(r'ATOM  ', l) != None
        elif input_format == "PUREMD":
            regex = lambda l: match(r'BOXGEO', l) == None

        with open(input_file, 'r') as infile:
            for line in infile:
                if regex(line):
                    count = count + 1

        return count


    patterns = { \
            ("BGF", "PDB"): [ \
            lambda l, **kwargs: ('{:6}{:5d} {:>4}{:1}{:3} {:1}{:4d}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4}{:>2}{:2}'.format(
                        'ATOM  ', int(l[7:12].strip()), l[13:18].strip(),
                        ' ', '   ', ' ', 0, ' ',
                        float(l[32:42].strip()),
                        float(l[42:52].strip()),
                        float(l[52:62].strip()),
                        1.0, 0.0, '    ', l[61:66].strip(), '  ') + linesep, True)
                    if not match(r'HETATM ', l) == None else (l, False),
                lambda l, **kwargs: (l, True),
                    ],
            ("BGF", "PUREMD"): [ \
                #TODO
                    ],
            ("PDB", "BGF"): [ \
                lambda l, **kwargs: ('HETATM {:5d} {:<5} {:3} {:1} {:5}{:10.5f}{:10.5f}{:10.5f} {:5}{:3d}{:2d} {:8.5f}'.format(
                        int(l[6:11].strip()), l[76:78].strip(), '  ', ' ', '     ', float(l[30:38].strip()),
                        float(l[38:46].strip()), float(l[46:54].strip()), l[76:78].strip(),
                        1, 1, 0.0) + linesep, True)
                    if not match(r'ATOM  ', l) == None else (l, False),
                lambda l, **kwargs: (l, True),
                                    ],
            ("PDB", "PUREMD"): [ \
                lambda l, **kwargs: (sub(r'^ATOM  ([ 0-9]{5}) (.{4}).{14}([. 0-9]{8})([. 0-9]{8})([. 0-9]{8}).{22}([ a-zA-z]{2}).{2}$',
                    r'\1 \2 \6 \3 \4 \5', l), True) if not match(r'ATOM  ', l) == None else (l, False),
                lambda l, **kwargs: (sub(r'^CRYST1([. 0-9]{9})([. 0-9]{9})([. 0-9]{9})([. 0-9]{7})([. 0-9]{7})([. 0-9]{7}) .{15}$',
                    r'BOXGEO \1 \2 \3 \4 \5 \6', l), False) if not match(r'CRYST1', l) == None else (l, False),
                lambda l, **kwargs: (l + str(kwargs['atom_count']) + linesep, True)
                    if not match(r'BOXGEO', l) == None else (l, False)
            ],
            ("PUREMD", "PDB"): [ \
                #TODO
                    ],
            ("PUREMD", "BGF"): [ \
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
    from re import match
    from os import linesep

    if args.input_format == 'BGF':
        #TODO
        pass

    elif args.input_format == 'PDB':
        cryst_regex = lambda l: match(r'^CRYST1', l) != None
        atom_regex = lambda l: match(r'ATOM  ', l) != None

        atom_lines = []
        cryst_line = ''
        x_dim = 0.0
        y_dim = 0.0
        z_dim = 0.0
        with open(args.input_file, 'r') as infile:
            for line in infile:
                if cryst_regex(line):
                    cryst_line = line
                    line = line.split()
                    x_dim = float(line[1])
                    y_dim = float(line[2])
                    z_dim = float(line[3])
                elif atom_regex(line):
                    atom_lines.append(line)

        out_file = 'output.pdb'
        if args.output_file:
            out_file = args.output_file

        with open(out_file, 'w') as outfile:
            # box dimensions
            outfile.write('{:6}{:9.3f}{:9.3f}{:9.3f}{:7}{:7}{:7} {:11}{:4}'.format(
                'CRYST1', x_dim * args.X_repl, y_dim * args.Y_repl, z_dim * args.Z_repl,
                cryst_line[33:40], cryst_line[40:47], cryst_line[47:54],
                cryst_line[54:65], cryst_line[65:69]) + linesep)

            # atoms
            count = 1
            for x in range(args.X_repl):
                for y in range(args.Y_repl):
                    for z in range(args.Z_repl):
                        for l in atom_lines:
                            outfile.write('{:6}{:5d} {:>4}{:1}{:3} {:1}{:4d}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4}{:>2}{:2}'.format(
                                'ATOM  ', count, l[12:16].strip(),
                                ' ', '   ', ' ', 0, ' ',
                                float(l[30:38].strip()) + x * x_dim,
                                float(l[38:46].strip()) + y * y_dim,
                                float(l[46:54].strip()) + z * z_dim,
                                1.0, 0.0, l[72:76], l[76:78].strip(), '  ') + linesep)

                            count = count + 1

    elif args.input_format == 'PUREMD':
        atom_count = 0
        atom_lines = []
        # simulation box info
        x_dim = 0.0
        y_dim = 0.0
        z_dim = 0.0
        x_ang = 0.0
        y_ang = 0.0
        z_ang = 0.0

        with open(args.input_file, 'r') as infile:
            # BOXGEO line
            line = infile.readline()
            line = line.split()
            x_dim = float(line[1])
            y_dim = float(line[2])
            z_dim = float(line[3])
            x_ang = float(line[4])
            y_ang = float(line[5])
            z_ang = float(line[6])

            # atom count line
            line = infile.readline()
            line = line.split()
            atom_count = int(line[0])

            # atom info lines
            for line in infile:
                atom_lines.append(line.split())

        if args.output_file:
            out_file = args.output_file
        else:
            out_file = 'output.geo'

        with open(out_file, 'w') as outfile:
            # BOXGEO line
            outfile.write('{:6} {:f} {:f} {:f} {:f} {:f} {:f}'.format(
                'BOXGEO', x_dim * args.X_repl, y_dim * args.Y_repl, z_dim * args.Z_repl,
                x_ang, y_ang, z_ang) + linesep)

            # atom count line
            outfile.write('{:d}'.format(atom_count * args.X_repl * args.Y_repl * args.Z_repl) + linesep)

            # atom info lines
            count = 1
            for x in range(args.X_repl):
                for y in range(args.Y_repl):
                    for z in range(args.Z_repl):
                        for l in atom_lines:
                            outfile.write(' {:d} {:2} {:2} {:f} {:f} {:f}'.format(
                                count, l[1], l[2],
                                float(l[3]) + x * x_dim,
                                float(l[4]) + y * y_dim,
                                float(l[5]) + z * z_dim ) + linesep)

                            count = count + 1


def noise(args):
    from numpy.random import normal
    from re import match
    from os import linesep

    mu, sigma = 0.0, args.sigma

    if args.input_format == 'BGF':
        #TODO
        pass

    elif args.input_format == 'PDB':
        atom_regex = lambda l: match(r'ATOM  ', l) != None

        out_file = 'output.pdb'
        if args.output_file:
            out_file = args.output_file

        with open(args.input_file, 'r') as infile, open(out_file, 'w') as outfile:
            for line in infile:
                if atom_regex(line):
                    outfile.write('{:6}{:5d} {:>4}{:1}{:3} {:1}{:4d}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4}{:>2}{:2}'.format(
                        'ATOM  ', int(line[6:12].strip()), line[12:16].strip(),
                        ' ', '   ', ' ', 0, ' ',
                        float(line[30:38].strip()) + normal(mu, sigma),
                        float(line[38:46].strip()) + normal(mu, sigma),
                        float(line[46:54].strip()) + normal(mu, sigma),
                        1.0, 0.0, line[72:76], line[76:78].strip(), '  ') + linesep)
                else:
                    outfile.write(line)

    elif args.input_format == 'PUREMD':
        #TODO
        pass


if __name__ == "__main__":
    import argparse

    FORMATS=["BGF", "PDB", "PUREMD"]

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
    parser_replicate.add_argument('X_repl', metavar='X', type=int,
            help='replication factor in X dimension')
    parser_replicate.add_argument('Y_repl', metavar='Y', type=int,
            help='replication factor in Y dimension')
    parser_replicate.add_argument('Z_repl', metavar='Z', type=int,
            help='replication factor in Z dimension')
    parser_replicate.add_argument('output_file', metavar='out_file', nargs='?',
            help='output geometry file')
    parser_replicate.set_defaults(func=replicate)

    # noise subcommand
    parser_noise = subparsers.add_parser('noise', aliases=['n', 'no'],
            help='Create new geometry from existing by adding prescribed noise to atomic positions.')
    parser_noise.add_argument('input_format', metavar='in_format',
            choices=FORMATS, help='input format')
    parser_noise.add_argument('input_file', metavar='in_file',
            help='input geometry file')
    parser_noise.add_argument('sigma', metavar='sigma', type=float,
            help='standard deviation of the normal distribution used to modeling the added noise to atomic position')
    parser_noise.add_argument('output_file', metavar='out_file', nargs='?',
            help='output geometry file')
    parser_noise.set_defaults(func=noise)

    # parse args and take action
    args = parser.parse_args()
    args.func(args)
