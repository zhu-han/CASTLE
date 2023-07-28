#!/usr/bin/python

import argparse
from ast import arg
import os
import shutil

def get_parser():
    parser = argparse.ArgumentParser(
        description='Combine decode results from parallel processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', type=str, help='decode data dir before split')
    parser.add_argument('--decodedir', type=str, help='decode result dir after combine')
    parser.add_argument('--subset', type=str, help='decode subset name')
    parser.add_argument('--num', type=int, help='number of line to split')
    parser.add_argument('--files', type=str, help='the target files')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    split_datadir = os.path.join(args.datadir, "split" + str(args.num))
    split_decodedir = os.path.join(args.decodedir, "split" + str(args.num))
    
    total_tsv_file = os.path.join(args.datadir, args.subset + ".tsv")
    with open(total_tsv_file, "r") as fr:
        lines = fr.readlines()

    # id_list is the list of file_name in tsv
    id_list = []
    for line in lines[1:]:
        id_list.append(line.split(".")[0].split("/")[-1])

    files = args.files.split(",")
    for file in files:
        file_replace = {}
        file_doc = ""
        total_decode_file = os.path.join(args.decodedir, file)

        for i in range(args.num):
            split_tsv_file = os.path.join(split_datadir, args.subset + "_" + str(i+1) + ".tsv")
            split_decode_file = os.path.join(split_decodedir, file.replace(".txt", "") + "_" + str(i+1) + ".txt")
            with open(split_tsv_file, "r") as fr:
                lines = fr.readlines()
            # split_id_list is the list of file_name in tsv
            split_id_list = []
            for line in lines[1:]:
                split_id_list.append(line.split(".")[0].split("/")[-1])
            with open(split_decode_file, "r") as fr:
                lines = fr.readlines()
            for line in lines:
                # for transcription files
                if "(" in line:
                    id_old = line.split("(")[1].split(")")[0]
                # for score files
                else:
                    id_old = line.split("\t")[0]
                idx_old = int(id_old.split("-")[1])
                # idx_old is the index and id_new is the file_name
                id_new = split_id_list[idx_old]
                file_replace[id_new] = line.replace(id_old, id_new)
        for idx, id in enumerate(id_list):
            file_doc += file_replace[id].replace(id, "None-" + str(idx))
        with open(total_decode_file, "w") as fw:
            fw.write(file_doc)
