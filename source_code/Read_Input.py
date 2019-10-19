import csv

#This function reads the tsv file given by the file_name parameter
def read_tsv(file_name, separator):

    try:
        f = open(file_name, 'r', newline='', encoding='utf-8')
    except IOError:
        print('Cannot open the file <{}>'.format(file_name))
        raise SystemExit
    tsv_read = csv.reader(f, delimiter=separator)

    return tsv_read

#This function separates the tweets and labels into two different lists and also balance data.
def process_data(data, balanced_data = False, test_data=False):

    # tweets and class labels will be stored here
    tweets = []
    labels = []

    data = list(data)
    if test_data:
        for line in data:
            tweet, _, _, _ = line
            # making multiple line tweet to a single line
            tweet = tweet.replace('\n', ' ').replace('\r', ' ')
            tweets.append(tweet)

        return tweets

    else:
        for line in data:
            tweet, _, _, label, _ = line
            # making multiple line tweet to a single line
            tweet = tweet.replace('\n', ' ').replace('\r', ' ')
            # if the following check is right then the parsing is right
            if label not in ['OFF', 'NOT']:
                # if any of the class label is not right
                print('Parsing Error of Class Label at {}'.format(id))

            else:
                # For Balancing Training data..makes same number of samples from two different classes
                if label != 'OFF' and balanced_data and labels.count("NOT")>=3400:
                    continue

                tweets.append(tweet)
                labels.append(label)
        return tweets, labels