# import pickle
#
# text = pickle.load(open('finished_files_twitter_inconsistent_3m10/extract_info.pkl', 'rb'), encoding='utf-8')
# print(len(text.get('finished_files_twitter_inconsistent_3m10/train').get('extract_words_num')))
import os
import random

nums = []

def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

def get_random_data(arr):
    for text_file in arr:
        highlights = []
        twitter = []
        next_is_highlight = False
        w = open(text_file.replace(text_file.split('/')[0], 'twitter_random'), "w", encoding="utf-8")
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                elif line.startswith("@highlight"):
                    next_is_highlight = True
                elif next_is_highlight:
                    highlights.append(line)
                    next_is_highlight = False
                elif line != "":
                    twitter.append(line)
        random.shuffle(twitter)
        for i in range(len(twitter)):
            w.write(twitter[i] + '\n')
            w.write('\n')
        for i in range(len(highlights)):
            w.write('@highlight' + '\n')
            w.write('\n')
            w.write(highlights[i] + '\n')
            w.write('\n')

if __name__ == "__main__":
    # 根目录路径
    root_path = r"twitter_final_3m10"
    # 用来存放所有的文件路径
    file_list = []
    # 用来存放所有的目录路径
    dir_list = []
    get_file_path(root_path, file_list, dir_list)
    print(file_list)
    get_random_data(file_list)
    # print(file_list)
    # highlight = read_text_file(file_list)
    # word_nums = 0
    # print(len(highlight))
    # for line in highlight:
    #     lines = line.split(' ')
    #     print(len(lines))
    #     word_nums += len(lines)
    # # print(dir_list)
    # print(word_nums)
