#!/usr/bin/python

import argparse
import os
import shutil

def get_parser():
    parser = argparse.ArgumentParser(
        description='Split decode data dir for parallel processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', type=str, help='decode data dir to split')
    parser.add_argument('--subset', type=str, help='decode subset name')
    parser.add_argument('--num', type=int, help='number of line to split')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    tsv_file = os.path.join(args.datadir, args.subset + ".tsv")
    ltr_file = os.path.join(args.datadir, args.subset + ".ltr")
    split_dir = os.path.join(args.datadir, "split" + str(args.num))
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.mkdir(split_dir)
    for file in [tsv_file, ltr_file]:
        with open(file, "r") as fr:
            lines = fr.readlines()
        if file.endswith("tsv"):
            root = lines[0]
            lines = lines[1:]
        else:
            root = None

        num_per_file = (len(lines) // args.num) + 1
        for i in range(args.num):
            base_file = os.path.basename(file)
            outname_i = os.path.join(split_dir, base_file.split(".")[0] + "_" + str(i+1) + "." + base_file.split(".")[1])
            with open(outname_i, "w") as fw:
                if root is not None:
                    fw.write(root)
                begin_pos = i * num_per_file
                end_pos = min((i + 1) * num_per_file, len(lines))
                for j in range(begin_pos, end_pos):
                    fw.write(lines[j])