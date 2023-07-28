#!/usr/bin/python3
import argparse
import math

def trn2text(threshold, uncertainty_weight, length_bonus, trn, wrd, ltr, tsv):
    with open(trn, "r") as fr:
        lines = fr.readlines()
    hypo = []
    for line in lines:
        line_list = line.strip().split("\t")
        if len(line_list) == 3:
            name, dropout, score = line_list
            content = ""
        elif len(line_list) == 4:
            name, dropout, score, content = line_list
        else:
            raise RuntimeError(f"This line not right: ", line)
        content = content.strip()
        name = name.strip().split("-")[1]
        hypo.append((int(name), float(score) - float(uncertainty_weight) *(float(dropout) - float(length_bonus) * math.log(len(content.split())+1)) , content))
    length = len(hypo)
    hypo = sorted(hypo, key=lambda r:r[0])
    wrd_text = []
    ltr_text = []
    kept_id = set()
    for line in hypo:
        if line[1] >= float(threshold):
            wrd_text.append(line[-1])
            ltr_text.append(" ".join(list(line[-1].replace(" ", "|"))) + " |")
            kept_id.add(line[0])

    wrd_doc = "\n".join([line for line in wrd_text])
    ltr_doc = "\n".join([line for line in ltr_text])
    with open(wrd, "w") as fw:
        fw.write(wrd_doc)
    with open(ltr, "w") as fw:
        fw.write(ltr_doc)
    with open(tsv, "r") as fr:
        lines = fr.readlines()
    head = lines.pop(0).strip()
    tsv_list = [head]
    for idx, line in enumerate(lines):
        if idx in kept_id:
            tsv_list.append(line.strip())

    tsv_doc = "\n".join([line for line in tsv_list])
    with open(tsv, "w") as fw:
        fw.write(tsv_doc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold")
    parser.add_argument("--uncertainty-weight")
    parser.add_argument("--length-bonus")
    parser.add_argument("--trn")
    parser.add_argument("--wrd")
    parser.add_argument("--ltr")
    parser.add_argument("--tsv")
    args = parser.parse_args()
    trn2text(args.threshold, args.uncertainty_weight, args.length_bonus, args.trn, args.wrd, args.ltr, args.tsv)
