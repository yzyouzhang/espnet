# -*- coding: UTF-8 -*-

from xml.dom.minidom import parse
from argparse import ArgumentParser
import shutil
import string
import os
import sys
import re

s = ''.join(chr(c) for c in range(sys.maxunicode+1))
ws = ''.join(re.findall(r'\s', s))
outtab = " " * len(ws)
trantab = str.maketrans(ws, outtab)
delset = string.punctuation
delset = delset.replace(":", "")
delset = delset.replace("'", "")

def _ToChar(text):
    new_text = ""
    for char in text:
        if char == " ":
            new_text += "<SPACE>"
        else:
            new_text += char
        new_text += " "
    return new_text

def TextRefine(text):
    text = re.sub("\.\.\.|\*|\[.*?\]", "", text.upper())
    delset_specific = delset
    remove_clear = "()=-"
    for char in remove_clear:
        delset_specific = delset_specific.replace(char, "")
    return text.translate(str.maketrans("", "", delset_specific))


def ExtractAudioID(audioname, wav_spk_info=None):
    if wav_spk_info:
        for key in wav_spk_info:
            if key in audioname:
                return key
    else:
      print("ERROR in audioname")
    return "error"


def PackZero(number, size=6):
    return "0" * (size - len(str(number))) + str(number)


def LoadWavSpeakerInfo(info_file):
    '''return dict of wav: spk_list'''

    info_file = open(info_file, "r", encoding="utf-8")
    return list(info_file.read().split("\n"))


def TimeOrderProcess(time_order_dom):
    time_order = {}
    time_slots = time_order_dom.getElementsByTagName("TIME_SLOT")
    for time_slot in time_slots:
        # convert to second based
        time_order[time_slot.getAttribute("TIME_SLOT_ID")] = float(time_slot.getAttribute("TIME_VALUE")) / 1000
    return time_order


def ELANProcess(afile, langs):
    try:
        elan_content = parse(afile).documentElement
    except:
        print("encoding failed  %s, the eaf file has some format issues" % afile)
        return None
    time_order = TimeOrderProcess(elan_content.getElementsByTagName("TIME_ORDER")[0])
    tiers = elan_content.getElementsByTagName("TIER")
    channels = {}
    for tier in tiers:
        if tier.getAttribute("LINGUISTIC_TYPE_REF") not in ["ASR"] :
            # only consider pure caption
            continue

        annotations = tier.getElementsByTagName("ANNOTATION")
        for anno in annotations:
            info = anno.getElementsByTagName("ALIGNABLE_ANNOTATION")[0]
            seg_id = info.getAttribute("ANNOTATION_ID")
            start = time_order[info.getAttribute("TIME_SLOT_REF1")]
            end = time_order[info.getAttribute("TIME_SLOT_REF2")]
            if start == end:
                continue
            text = ""
            childs = info.getElementsByTagName("ANNOTATION_VALUE")[0].childNodes
            for child in childs:
                if child.firstChild is not None:
                    continue
                    text += child.firstChild.data
                else:
                    text += child.data
            if langs == "mixtec" and "$" in text:
                continue
            elif langs == "spanish" and "$" not in text:
                continue
            else:
                text = "blank"
            channels[seg_id[1:]] = [start, end, text]
    return channels


def TraverseData(annotation_dir, sound_dir, target_dir, speaker_info, langs):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    text = open(os.path.join(target_dir, "text"), "w", encoding="utf-8")

    # get relationship
    sound_files = {}
    annotation_files = {}
    wav_spk_info = LoadWavSpeakerInfo(speaker_info)

    for root, dirs, files in os.walk(sound_dir):
            for file in files:
                if file[-4:] == ".wav":
                    sound_files[ExtractAudioID(file, wav_spk_info)] = os.path.join(root, file)
        
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file[-4:] == ".eaf":
                annotation_files[ExtractAudioID(file, wav_spk_info)] = os.path.join(root, file)
    
    for afile in annotation_files.keys():
        afile_path = annotation_files[afile]
        print(afile)
        if afile == "error":
            continue
        if afile not in sound_files.keys():
            print("not found wav files for {}".format(afile))
            continue
        segment_info = ELANProcess(afile_path, langs)
        if segment_info is None or segment_info == {}:
            print("no segment found for {}\n".format(afile) + "Please make sure the TIER's name is set to 'ASR'")
            continue

        print("%s sox -t wavpcm \"%s\" -c 1 -r 16000 -b 16 -t wavpcm - |" % (afile, sound_files[afile]), file=wavscp)

        for segment_number, segment in segment_info.items():
            segment_id = "%s_%s" % (afile, PackZero(segment_number))
            print("%s %s %s %s" %(segment_id, afile, segment[0], segment[1]), file=segments)
            print("%s %s" %(segment_id, segment_id), file=utt2spk)
            print("%s %s" %(segment_id, segment_id), file=spk2utt)
            print("%s %s" %(segment_id, _ToChar(segment[2])), file=text)

        print("successfully processing %s" % afile)

    segments.close()
    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    text.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='Process Raw data with correction')
    parser.add_argument('-a', dest="ann_path", type=str, help="annotation path", default="/export/c04/jiatong/data/Yoloxochitl-Mixtec-for-ASR/Field-biology-mono-new-transcription")
    parser.add_argument('-t', dest="target_dir", type=str, help='target_dir', default="data/botany_mono_121_blank_mixtec_only")
    parser.add_argument('-s', dest="sound_dir", type=str, help="sound dir", default="/export/c04/jiatong/data/Yoloxochitl-Mixtec-for-ASR/Field-biology-mono-new-transcription")
    parser.add_argument('-i', dest='speaker_info', type=str, help='speaker info file dir', default='local/blank_files.csv')
    parser.add_argument('--lang', type=str, help="language type", default="mixtec")
    args = parser.parse_args()
    TraverseData(args.ann_path, args.sound_dir, args.target_dir, speaker_info=args.speaker_info, langs=args.lang)
