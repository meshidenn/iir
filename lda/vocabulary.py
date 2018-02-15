
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import nltk
import re
import MeCab
import glob


def load_corpus(ranges):
    """
    load data from corpus
    """
    tmp = re.match(r'(\d+):(\d+)$', ranges)
    if tmp:
        start = int(tmp.group(1))
        end = int(tmp.group(2))
        from nltk.corpus import brown as corpus
        return [corpus.words(fileid) for fileid in corpus.fileids()[start:end]]


def load_file(filename):
    """
    for one file
    one line corresponds to one doc
    """
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?', line)
        if len(doc) > 0:
            corpus.append(doc)
    f.close()
    return corpus


def load_file_ja(dirname):
    """
    for multiple file
    one file corresponds to one doc
    target file is directed by target
    On default, it is dir/*.txt
    """
    corpus = []
    tagger = MeCab.Tagger('-Ochasen')
    tagger.parse("")
    target = dirname + '*.txt'
    files = glob.glob(target)
    pos = ['名詞', '形容詞', '動詞']
    for fn in files:
        print(fn)
        with open(fn, 'r') as f:
            sentences = []
            for line in f:
                node = tagger.parseToNode(line)
                while node:
                    if node.feature.split(',')[0] in pos:
                        # print(node.surface)
                        sentences.append(node.surface)
                    node = node.next
        corpus.append(sentences)
    return corpus, files
# stopwords_list = nltk.corpus.stopwords.words('english')

stopwords_list = "する,.,:,いる,!,-,/,し,さ,の,こと,れ,さん,なる,なっ,ある,い,できる,ーー,ー,よう,め,たち,そう,いう,ちゃん,://,com,news,年,歳,...」,cl,!!,!!」（,!!　,%,•,ッ,アノ,ない,人,者,もの,かな,t,的,http,livedoor,くれ,とき,かな,ん,それ,あり,ため,),⇔".split(',')
recover_list = {"wa":"was", "ha":"has"}
wl = nltk.WordNetLemmatizer()


def is_stopword(w):
    return w in stopwords_list


def lemmatize(w0):
    w = wl.lemmatize(w0.lower())
    # if w=='de': print w0, w
    if w in recover_list: return recover_list[w]
    return w


class Vocabulary:
    def __init__(self, excluds_stopwords=False):
        self.vocas = []         # id to word
        self.vocas_id = dict()  # word to id
        self.docfreq = []       # id to document frequency
        self.excluds_stopwords = excluds_stopwords

    def term_to_id(self, term0):
        term = lemmatize(term0)
        # if not re.match(r'[a-z]+$', term): return None
        if self.excluds_stopwords and is_stopword(term): return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
            self.docfreq.append(0)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):
        # print ' '.join(doc)
        list = []
        words = dict()
        # print(doc)
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                if not id in words:
                    words[id] = 1
                    self.docfreq[id] += 1
        if "close" in dir(doc): doc.close()
        return list
    
    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list

