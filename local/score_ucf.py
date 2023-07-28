import editdistance
import math



def read_pseudo_label_file(file_name):
    pseudo_label = list()
    with open(file_name, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 4:
            line_list.append("")
        assert len(line_list) == 5
        pseudo_label.append(line_list)
    return pseudo_label


def score_top_confidence_pseudo_label(pseudo_label_file, reference_label_file, start_percent, end_percent):
    diff_pseudo_label = []
    reference_label = {}

    pseudo_label = read_pseudo_label_file(pseudo_label_file)

    with open(reference_label_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("(")
        if len(line_list) == 1:
            line_list.insert(0, "")
        assert len(line_list) == 2
        reference_label[line_list[1].split(")")[0]] = line_list[0].strip()

    for first in pseudo_label:
        index = first[0]
        wer = float(first[1])
        score = float(first[2])
        text = first[-1] 
        diff_pseudo_label.append([index, score - 10 *  (wer - 0.1 * math.log(len(text.split())+1)), text])
    label_length = len(diff_pseudo_label)
    diff_pseudo_label = sorted(diff_pseudo_label, key=lambda r:r[1], reverse=True)[int(label_length * float(start_percent)):int(label_length * float(end_percent))]
    total_err = 0
    total_len = 0
    for items in diff_pseudo_label:
        err = editdistance.eval(items[-1].split(), reference_label[items[0]].split())
        length = len(reference_label[items[0]].split())
        if True:
            total_err += err
            total_len += length
    print("Threshold: ", diff_pseudo_label[-1][1])
    print("WER: ", total_err / total_len)

if __name__ == "__main__":
    score_top_confidence_pseudo_label("hypo.word.score-checkpoint_best.pt-dev.txt", "ref.word-0-checkpoint_best.pt-dev.txt", 0.0, 0.5)
    
