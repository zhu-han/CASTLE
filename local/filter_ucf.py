import editdistance
import argparse
import os

def score_top_confidence_pseudo_label(pseudo_label_0_file, pseudo_label_1_file, pseudo_label_2_file, pseudo_label_3_file, percent, output_file):
    pseudo_label_0 = []
    pseudo_label_1 = []
    pseudo_label_2 = []
    pseudo_label_3 = []
    diff_pseudo_label = []
    with open(pseudo_label_0_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 2:
            line_list.append("")
        assert len(line_list) == 3
        pseudo_label_0.append(line_list)

    with open(pseudo_label_1_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 2:
            line_list.append("")
        assert len(line_list) == 3
        pseudo_label_1.append(line_list)

    with open(pseudo_label_2_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 2:
            line_list.append("")
        assert len(line_list) == 3
        pseudo_label_2.append(line_list)

    with open(pseudo_label_3_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 2:
            line_list.append("")
        assert len(line_list) == 3
        pseudo_label_3.append(line_list)

    for first, second, thrid, fourth in zip(pseudo_label_0, pseudo_label_1, pseudo_label_2, pseudo_label_3):
        errs = [editdistance.eval(second[2].split(), first[2].split()), editdistance.eval(thrid[2].split(), first[2].split()), editdistance.eval(fourth[2].split(), first[2].split())]
        length = len(first[2].split())
        wers = []
        for err in errs:
            wers.append(err / length if length != 0 else err)
        wer = sum(wers) / len(wers)
        index = first[0]
        score = first[1]
        text = first[2] 
        diff_pseudo_label.append([index, wer, score, text])
    diff_pseudo_label = sorted(diff_pseudo_label, key=lambda x: float(x[1]), reverse=False)[:int(len(diff_pseudo_label) * percent)]
    diff_pseudo_label_doc = ""
    for items in diff_pseudo_label:
        diff_pseudo_label_doc += "\t".join([str(i) for i in items])
        diff_pseudo_label_doc += "\n"
    with open(output_file, "w") as fw:
        fw.write(diff_pseudo_label_doc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()

    score_top_confidence_pseudo_label(os.path.join(args.dir, "hypo.word.score-0-checkpoint_best.pt-train.txt"), os.path.join(args.dir, "hypo.word.score-1-checkpoint_best.pt-train.txt"), os.path.join(args.dir, "hypo.word.score-2-checkpoint_best.pt-train.txt"), os.path.join(args.dir, "hypo.word.score-3-checkpoint_best.pt-train.txt"), 1.0, os.path.join(args.dir, "hypo.word.score-checkpoint_best.pt-train.txt"))
