import functools
import operator
import emoji
import string

#This function reads the modified stopword list from the project source directory
def read_stopwords():
    stop_list = []

    with open('stoplist.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_list.append(line.strip())

    return stop_list


#this function splits emoji and return word tokens while multiple emojis are separated as single tokens
def make_tokens_with_split_emoji(line):
    # insert code for adding whitespace to emojis here
    em_split_emoji = emoji.get_emoji_regexp().split(line)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split_tokens = functools.reduce(operator.concat, em_split_whitespace)
    return em_split_tokens


# Tokenize and Append the text in documents array.
def tokenize(data):

    # try:
    #     # modified stopword list from file
    #     stopWords = set(read_stopwords())
    # except:
    #     print("Stop Word List Not Found! ")

    documents = []
    # tokenize the lines without stopwords
    for line in data:

        documents.append(make_tokens_with_split_emoji(line))
        # tokenize the lines with stopwords
        # line = make_tokens_with_split_emoji(line)
        # doc = []
        # for token in line:
        #     if token.lower() not in stopWords:
        #         doc.append(token)
    return documents

# Show Class Distribution of Data
def distribution(classLabels, title=""):

    classLabels = list(classLabels)
    labels = list(set(classLabels))
    count_class = [0] * len(labels)
    index = 0
    for label in labels:
        count_class[index] = classLabels.count(label)
        index += 1

    print("\nDistribution of classes in {} Set:".format(title))
    print(labels)
    print(count_class)
    print()