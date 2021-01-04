from argparse import ArgumentParser
from re import findall, sub, S
import os


def ELANFormat(data_dir, hyp_file, output_dir):
    hyp_info = {}
    with open(hyp_file, "r", encoding="utf-8") as hyp:
        hyps = hyp.read().strip()
        # utts = findall("\(.*?-(.*?)\)", hyps, S)
        hyps = hyps.split("\n")
        for hyp_line in hyps:
            for i in range(len(hyp_line)):
                if hyp_line[i] == "(" and "(" not in hyp_line[i + 1:]:
                    hyp_place = i
                    break
            hyp_info["-".join(hyp_line[hyp_place + 1: -1].split("-")[1:])] = (hyp_line[:hyp_place]).lower()
    seg_info = {}
    with open(os.path.join(data_dir, "segments"), "r", encoding="utf-8") as segments:
        segs = segments.read().strip().split("\n")
        for seg in segs:
            details = seg.split(" ")
            # remove -L -R if used
            # details[1] = details[1][:-2]
            # spk = details[0].split("_")[0]
            spk = "unknown"
            seg_info[details[1]] = seg_info.get(details[1], [])
            seg_info[details[1]].append([details[0], spk, details[2], details[3]])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(hyp_info)
    print("*" * 20)
    print(seg_info)
    for seg in seg_info.keys():
        output_file = open(os.path.join(output_dir, "%s" %seg), "w", encoding="utf-8")
        info = seg_info[seg]
        for utt in info:
            output_file.write("%s\t%s\n" %("\t".join(utt[1:]), hyp_info[utt[0]]))
        output_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='kaldi data dir')
    parser.add_argument('result_files', type=str, help='result files')
    parser.add_argument('output_dir', type=str, help='output dir')
    args = parser.parse_args()
    ELANFormat(args.data_dir, args.result_files, args.output_dir)
