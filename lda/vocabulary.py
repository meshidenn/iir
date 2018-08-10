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

def load_dataframe(documents):
    corpus = []
    for doc in documents:
        sentences = re.findall(r'\w+(?:\'\w+)?', doc)
        if len(sentences) > 0:
            corpus.append(sentences)

    return corpus

def load_corpus_ja(documents):
    corpus = []
    tagger = MeCab.Tagger('-Ochasen')
    tagger.parse("")
    pos = ['名詞', '形容詞', '動詞']
    for doc in documents:
        sentences = []
        node = tagger.parseToNode(doc)
        while node:
            if node.feature.split(',')[0] in pos:
                # print(node.surface)
                sentences.append(node.surface)
            node = node.next
    corpus.append(sentences)
    return corpus


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



"""
stopwords_list = "する,.,:,いる,!,-,/,し,さ,の,こと,れ,さん,なる,なっ,ある,い,できる,ーー,ー,よう,め,たち,そう,いう,ちゃん,://,com,news,年,歳,...」,cl,!!,!!」（,!!　,%,•,ッ,アノ,ない,人,者,もの,かな,t,的,http,livedoor,くれ,とき,かな,ん,それ,あり,ため,),⇔".split(',')
"""
stopwords_list = "あそこ,あたり,あちら,あっち,あと,あな,あなた,あれ,いくつ,いつ,いま,いや,いろいろ,うち,おおまか,おまえ,おれ,がい,かく,かたち,かやの,から,がら,きた,くせ,ここ,こっち,こと,ごと,こちら,ごっちゃ,これ,これら,ごろ,さまざま,さらい,さん,しかた,しよう,すか,ずつ,すね,すべて,ぜんぶ,そう,そこ,そちら,そっち,そで,それ,それぞれ,それなり,たくさん,たち,たび,ため,だめ,ちゃ,ちゃん,てん,とおり,とき,どこ,どこか,ところ,どちら,どっか,どっち,どれ,なか,なかば,なに,など,なん,はじめ,はず,はるか,ひと,ひとつ,ふく,ぶり,べつ,へん,ぺん,ほう,ほか,まさ,まし,まとも,まま,みたい,みつ,みなさん,みんな,もと,もの,もん,やつ,よう,よそ,わけ,わたし,ハイ,上,中,下,字,年,月,日,時,分,秒,週,火,水,木,金,土,国,都,道,府,県,市,区,町,村,各,第,方,何,的,度,文,者,性,体,人,他,今,部,課,係,外,類,達,気,室,口,誰,用,界,会,首,男,女,別,話,私,屋,店,家,場,等,見,際,観,段,略,例,系,論,形,間,地,員,線,点,書,品,力,法,感,作,元,手,数,彼,彼女,子,内,楽,喜,怒,哀,輪,頃,化,境,俺,奴,高,校,婦,伸,紀,誌,レ,行,列,事,士,台,集,様,所,歴,器,名,情,連,毎,式,簿,回,匹,個,席,束,歳,目,通,面,円,玉,枚,前,後,左,右,次,先,春,夏,秋,冬,一,二,三,四,五,六,七,八,九,十,百,千,万,億,兆,下記,上記,時間,今回,前回,場合,一つ,年生,自分,ヶ所,ヵ所,カ所,箇所,ヶ月,ヵ月,カ月,箇月,名前,本当,確か,時点,全部,関係,近く,方法,我々,違い,多く,扱い,新た,その後,半ば,結局,様々,以前,以後,以降,未満,以上,以下,幾つ,毎日,自体,向こう,何人,手段,同じ,感じ".split(",")

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
