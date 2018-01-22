import sys
import MeCab
import argparse


def main1(args):
    infile = args.input
    outfile = args.output

    sentences = []
    with open(infile, 'r', encoding='utf-8') as f:
        for i,line in enumerate(f):
            m = MeCab.Tagger("-Ochasen")
            res = m.parse(line)
            print(i)
            sentence = []
            for row in res.split("\n"):
                word = row.split("\t")[0]
                if word =='EOS':
                    break
                sentence.append(word)

            sentences.append(sentence)

    print(sentences)


def main2(args):
    infile = args.input
    outfile = args.output

    sentences = []
    with open(infile, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            m = MeCab.Tagger("-Ochasen")
            m.parse("")
            node = m.parseToNode(line)
            print(i)
            sentence = []
            while node:
                if (node.surface != ''):
                    sentence.append(node.surface)
                node = node.next

            sentences.append(sentence)

    print(sentences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input argument')
    parser.add_argument("--input", '-i', help='input file pass')
    parser.add_argument('--output', '-o', help='output file pass')

    args = parser.parse_args()
    main2(args)

