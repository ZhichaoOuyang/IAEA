import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import numpy as np
import rouge_not_a_wrapper as my_rouge
import nltk
import json
from tqdm import tqdm
import pickle as pk
import pdb
from nltk.corpus import stopwords

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
# END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
#               ")"]  # acceptable ways to end a sentence

# tweet
END_TOKENS = ['\n']     # acceptable ways to end a sentence for tweet

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# all_train_urls = "./url_lists/all_train.txt"
# all_val_urls = "./url_lists/all_val.txt"
# all_test_urls = "./url_lists/all_test.txt"

tokenized_twitter_dir = "twitter_test_4m6"    # 分词后存的路径
# cnn_tokenized_stories_dir = "cnn_stories_tokenized"
# dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files_twitter_4m6"    # 生成的.bin .pkl文件存放的位置
chunks_dir = os.path.join(finished_files_dir, "chunked")    # 划分后存放的位置

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
# num_expected_cnn_stories = 92579
# num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 5  # num examples per chunk, for the chunked data , that is, a timeline

extract_sents_num = []
extract_words_num = []
article_sents_num = []

extract_info = {}

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def chunk_file(set_name):
    in_file = finished_files_dir + '/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    # for set_name in ['train', 'val', 'test']:
    for set_name in ['test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)    # 遍历
    stories.sort()
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            print("data filename:", s)
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', '-cp', '/data/chao/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):    # 加\n
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " \n"


def get_art_abs(story_file):     # 单个.story文件的处理
    global article_sents_num

    lines = read_text_file(story_file)     # 按行读取

    # Lowercase everything
    lines = [line.lower() for line in lines]   # 每个都处理成小写字母

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":    # 空行
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line.strip("\n"))

    # Make article into a single string
    # article = ' '.join(article_lines)

    # get extractive summary
    # article_sents = tokenizer.tokenize(article)
    article_sents = [a for a in article_lines]
    extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r = get_extract_summary(article_sents, highlights)

    article_sents_num.append(len(article_sents))

    return article_sents, highlights, extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2]
                d[p1+1][p2+1] = 'left'
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]
                d[p1+1][p2+1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    #print (numpy.array(d))
    s = []
    while m[p1][p2]:    #不为None时
        c = d[p1][p2]
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1-=1
            p2-=1
        if c =='left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    # print((''.join(s)))
    # print(s)
    return len(s)

def simplex_inconsistent_detect(article_sents):
    former_sents = article_sents[0:len(article_sents)-1]
    current_sent = article_sents[-1]
    words = stopwords.words('english')
    # print(words)
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
        words.append(w)
    disease_List = nltk.word_tokenize(current_sent)
    filtered_current_words = [word for word in disease_List if word not in words]
    for former_sent in former_sents:
        disease_List = nltk.word_tokenize(former_sent)
        filtered_fromer_words = [word for word in disease_List if word not in words]
        lcs_len = find_lcseque(filtered_current_words, filtered_fromer_words)
        # print(lcs_len)
        # print(len(filtered_current_words))
        # print(len(filtered_fromer_words))
        if lcs_len > max(len(filtered_current_words), len(filtered_fromer_words)) * 0.5:
            print("#######################################")
            return True
    return False




def get_extract_summary(article_sents, abstract_sents):
    if len(article_sents) == 0 or len(abstract_sents) == 0:
        return [], [], [], [], [], None

    global extract_sents_num
    global extract_words_num
    fscores = []
    precisions = []
    recalls = []
    for i, art_sent in enumerate(article_sents):    # 遍历一个article
        rouge_l_f, rouge_l_p, rouge_l_r = my_rouge.rouge_l_summary_level([art_sent], abstract_sents)  # 得到每个句子和abstract之间的rouge得分
        fscores.append(rouge_l_f)   # 依次放入一个数组中
        precisions.append(rouge_l_p)
        recalls.append(rouge_l_r)

    scores = np.array(recalls)   # 主要用recall来判断rouge
    sorted_scores = np.sort(scores)[::-1]   #取从后向前（相反）的元素,从大到小排序
    id_sort_by_scores = np.argsort(scores)[::-1]   # 得到的是索引，从大到小分数的索引
    max_Rouge_l_r = 0.0
    extract_ids = []
    extract_sents = []

    for i in range(len(article_sents)):
        new_extract_ids = sorted(extract_ids + [id_sort_by_scores[i]]) # 之前抽取的索引和现在的结合，看有没有比之前的大
        new_extract_sents = [article_sents[idx] for idx in new_extract_ids]
        _, _, Rouge_l_r = my_rouge.rouge_l_summary_level(new_extract_sents, abstract_sents)

        if Rouge_l_r > max_Rouge_l_r and simplex_inconsistent_detect(new_extract_sents) is False:
            extract_ids = new_extract_ids
            extract_sents = new_extract_sents
            max_Rouge_l_r = Rouge_l_r

    # for those articles that don't reach the 2 conditions
    if len(extract_sents) == 0:
        pdb.set_trace()
    extract_sents_num.append(len(extract_sents))
    extract_words = ' '.join(extract_sents).split(' ')
    extract_words_num.append(len(extract_words))
    return extract_sents, extract_ids, fscores, precisions, recalls, max_Rouge_l_r


def write_to_bin(out_file, makevocab=False):    # 将txt写成.bin的格式
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    # print("Making bin file for URLs listed in %s..." % url_file)
    # url_list = read_text_file(url_file)
    # url_hashes = get_url_hashes(url_list)
    # story_fnames = [s + ".story" for s in url_hashes]
    tokenized_twitter = os.listdir(tokenized_twitter_dir)    # 遍历
    tokenized_twitter.sort()
    for name in tokenized_twitter:
        print("write_to_bin: " , name)
    story_fnames = [name for name in tokenized_twitter]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    global extract_sents_num
    global extract_words_num
    global article_sents_num
    global extract_info
    extract_sents_num = []
    extract_words_num = []
    article_sents_num = []
    data = {'article': [], 'abstract': [], 'rougeLs': {'f': [], 'p': [], 'r': []}, \
            'gt_ids': [], 'select_ratio': [], 'rougeL_r': []}

    with open(out_file, 'wb') as writer:
        for idx, s in tqdm(enumerate(story_fnames)):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx) * 100.0 / float(num_stories)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(tokenized_twitter_dir, s)):     # 单个.story文件
                story_file = os.path.join(tokenized_twitter_dir, s)
            # elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            #     story_file = os.path.join(dm_tokenized_stories_dir, s)
            else:
                print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?")

            # Get the strings to write to .bin file
            article_sents, abstract_sents, extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r = get_art_abs(
                story_file)
            ratio = float(len(extract_sents)) / len(article_sents) if len(article_sents) > 0 else 0

            # save scores of all article sentences
            data['article'].append(article_sents)
            data['abstract'].append(abstract_sents)
            data['rougeLs']['f'].append(fs)
            data['rougeLs']['p'].append(ps)
            data['rougeLs']['r'].append(rs)
            data['gt_ids'].append(extract_ids)
            data['select_ratio'].append(ratio)
            data['rougeL_r'].append(max_Rouge_l_r)

            # Make abstract into a signle string, putting <s> and </s> tags around the sentences
            article = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in article_sents])
            abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_sents])
            extract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in extract_sents])
            extract_ids = ','.join([str(i) for i in extract_ids])

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode('utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode('utf-8')])
            tf_example.features.feature['extract'].bytes_list.value.extend([extract.encode('utf-8')])
            tf_example.features.feature['extract_ids'].bytes_list.value.extend([extract_ids.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                art_tokens = [t for t in art_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    with open(out_file[:-4] + '_gt.pkl', 'wb') as out:
        pk.dump(data, out)

    print(
    "Finished writing file %s\n" % out_file)
    print(
    'average extract sents num: ', float(sum(extract_sents_num)) / len(extract_sents_num))
    print(
    'average extract words num: ', float(sum(extract_words_num)) / len(extract_words_num))
    print(
    'average article sents num: ', float(sum(article_sents_num)) / len(article_sents_num))
    split_name = out_file.split('.')[0]
    extract_info[split_name] = {'extract_sents_num': extract_sents_num,
                                'extract_words_num': extract_words_num,
                                'article_sents_num': article_sents_num}

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding="utf-8") as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
        sys.exit()
    # cnn_stories_dir = sys.argv[1]
    # dm_stories_dir = sys.argv[2]
    twitter_dir = sys.argv[1]


    # Check the stories directories contain the correct number of .story files
    # check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    # check_num_stories(dm_stories_dir, num_expected_dm_stories)

    # Create some new directories
    if not os.path.exists(tokenized_twitter_dir): os.makedirs(tokenized_twitter_dir)
    # if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
    # if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    # tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
    # tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)
    tokenize_stories(twitter_dir, tokenized_twitter_dir)   # 得到分词后的txt

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(os.path.join(finished_files_dir, "test.bin"))
    # write_to_bin(os.path.join(finished_files_dir, "val.bin"))
    # write_to_bin(os.path.join(finished_files_dir, "train.bin"), makevocab=True)

    with open(os.path.join(finished_files_dir, 'extract_info.pkl'), 'wb') as output_file:
        pk.dump(extract_info, output_file)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()

