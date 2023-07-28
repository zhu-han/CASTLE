#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to create word and letter transcriptions
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx")
    parser.add_argument("--input-trn")
    parser.add_argument("--output-trn")
    args = parser.parse_args()

    index_list = []

    with open(args.input_trn, "r") as input_trn, open(args.idx, "r") as idx:
        idx_lines = idx.readlines()
        input_trn_doc = input_trn.read()
    for line in idx_lines:
        line = line.strip()
        index_list.append(line)
    for index in range(len(index_list)):
        input_trn_doc = input_trn_doc
        input_trn_doc = input_trn_doc.replace("(None-" + str(index) + ")", "(" + index_list[index] + ")").replace(" (", "(")
    with open(args.output_trn, "w") as output_trn:
        output_trn.write(input_trn_doc)



if __name__ == "__main__":
    main()
