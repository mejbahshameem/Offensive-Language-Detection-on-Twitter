import Read_Input
import Preprocess
import SVM_OL

def start_train_test():

    default_train_file = 'offensive_dataset/traintag.tsv'
    default_dev_file = 'offensive_dataset/devtag.tsv'
    default_test_file = 'offensive_dataset/testtag.tsv'

    # reads the raw dataset into a dataframe
    train_dataframe = Read_Input.read_tsv(default_train_file, separator='\t')
    dev_dataframe = Read_Input.read_tsv(default_dev_file, separator='\t')
    test_dataframe = Read_Input.read_tsv(default_test_file, separator='\t')

    # gets the tweets and class labels as list
    train_tweets_sentence, train_labels = Read_Input.process_data(train_dataframe, balanced_data=True)
    dev_tweets_sentence, dev_labels = Read_Input.process_data(dev_dataframe)
    test_tweets_sentence = Read_Input.process_data(test_dataframe, test_data=True)

    # tokenize the sentence into words
    train_tweets = Preprocess.tokenize(train_tweets_sentence)
    dev_tweets = Preprocess.tokenize(dev_tweets_sentence)
    test_tweets = Preprocess.tokenize(test_tweets_sentence)

    # show the distribution of data
    Preprocess.distribution(train_labels, "Training")
    Preprocess.distribution(dev_labels, "Development")

    # # Training the final SVM Model with only training data; uncomment it if you want to test the development data
    # classifier = SVM_OL.run_model(train_tweets, train_labels)

    # Running the model on the devset after training only with training data
    # title = 'Binary(OFF/NOT) + Linear SVM'
    # SVM_OL.classify_dev_set(classifier, dev_tweets, dev_labels, title)

    # Training the final SVM Model with training and development data
    classifier = SVM_OL.run_model(train_tweets+dev_tweets, train_labels+dev_labels)

    # Runing the model with actual Test Data (unlabelled)
    test_predicted_labels = SVM_OL.classify_test_set(classifier, test_tweets)
    Preprocess.distribution(test_predicted_labels, "Predicted Test Labels")

    # writing the predicted test labels to a file
    write_test_output(test_predicted_labels)


#Writes the test output to a file
def write_test_output(predicted_labels):

    filename = 'predictions.test'

    print('\nWriting the Test output on <{}>'.format(filename))

    with open(filename, 'w', encoding='utf-8') as file:
        for lbl in predicted_labels:
            file.write('{}\n'.format(lbl))


if __name__ == '__main__':
    start_train_test()