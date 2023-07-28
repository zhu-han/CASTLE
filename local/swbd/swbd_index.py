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
    parser.add_argument("--tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    index_list = []

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".idx"), "w"
        ) as idx_out:
        _ = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            wav_name = line.split("\t")[0].split("/")[-1].split(".")[0]
            spk_name = "_".join(wav_name.split("_")[:2]).replace("-", "_")
            print(spk_name + "-" + wav_name, file=idx_out)


if __name__ == "__main__":
    main()
