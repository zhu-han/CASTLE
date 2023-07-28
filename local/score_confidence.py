import editdistance

def score_top_confidence_pseudo_label(pseudo_label_file, reference_label_file, start_percent, end_percent):
    pseudo_label = []
    reference_label = {}
    with open(pseudo_label_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("\t")
        if len(line_list) == 3:
            line_list.append("")
        assert len(line_list) == 4
        pseudo_label.append(line_list)
    with open(reference_label_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("(")
        if len(line_list) == 1:
            line_list.insert(0, "")
        assert len(line_list) == 2
        reference_label[line_list[1].split(")")[0]] = line_list[0].strip()
    pseudo_label = sorted(pseudo_label, key=lambda x: float(x[1]), reverse=True)
    pseudo_label = pseudo_label[int(len(pseudo_label) * start_percent): int(len(pseudo_label) * end_percent)]
    total_err = 0
    total_len = 0
    total_frame_num = 0
    selected_num = 0
    for items in pseudo_label:
        if True:
            err = editdistance.eval(items[-1].split(), reference_label[items[0]].split())
            length = len(reference_label[items[0]].split())
            total_err += err
            total_len += length
            total_frame_num += float(items[2])
            selected_num += 1
    print("Average Frame Number: ", total_frame_num / selected_num)
    print("Average Label Length: ", total_len / selected_num)
    print("WER: ", total_err / total_len)

if __name__ == "__main__":
    score_top_confidence_pseudo_label("hypo.word.score-checkpoint_best.pt-train.txt", "ref.word-checkpoint_best.pt-train.txt", 0.0, 0.2)
    score_top_confidence_pseudo_label("hypo.word.score-checkpoint_best.pt-train.txt", "ref.word-checkpoint_best.pt-train.txt", 0.0, 1.0)
    score_top_confidence_pseudo_label("hypo.word.score-checkpoint_best.pt-train.txt", "ref.word-checkpoint_best.pt-train.txt", 0.8, 1.0)

