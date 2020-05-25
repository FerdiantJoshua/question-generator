import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file_path',
        default='',
        type=str,
        required=True,
        help='Path to input data file. The result will also be saved in the same directory as this file.',
    )
    args = parser.parse_args()

    with open(args.input_file_path, 'r') as f_in:
        lines = f_in.readlines()

    save_dir = os.path.dirname(args.input_file_path)
    base_name = os.path.basename(args.input_file_path).split('.')[0]
    with open(f'{save_dir}/{base_name}_source.txt', 'w') as f_out_src:
        with open(f'{save_dir}/{base_name}_target.txt', 'w') as f_out_tgt:
            for line in lines:
                splitted = line.split('<sep> ')
                f_out_src.write(' <sep> '.join(splitted[:-1]) + '<sep>\n')
                f_out_tgt.write(splitted[-1].strip().replace('</s>', '') + '\n')

if __name__ == '__main__':
    main()
