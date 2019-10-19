1. In our offensive_dataset directory we have the files(traintag.tsv, devtag.tsv, testtag.tsv) after running them
on CMU TweetNLP. So, it's necessary to use them as it is in that directory. We used relative path to mention these 
files location in the code. Please indicate these 3 files location correctly inside the file "Main.py" if you want to 
reproduce the result again.

2. You have to run the file "Main.py" to start the proceedings and keep other necessary files("Read_Input.py", "Preprocess.py", "SVM_OL.py", "Eval_Matrics.py", "stoplist.txt") to appropiate location preferably at the same place
as the "Main.py" resides inside the project to import those files without any errors.

3. The code section which tests the accuracy of development set is currently commented out as we are submitting the result 
on the test data. If you want to test the accuracy of our classifier on development data, then please just uncomment the code
of line number 31,34 and 35 of "Main.py" where only training data is used for making the model. Please comment out the line 
number 38, 41, 42 and 45 which is written for the purpose of action related to the test data. A confusion matrix will be generated
at the same location to give better understanding to the test prediction of the development data

4. For running the classifier on test data just run "Main.py" by following the instructions mentioned in point 1 and 2. You don't 
need to uncomment anything as we are submitting the file in a way so that predictions for test data can be run without uncommenting
any part of the code. A "predictions.test" file will be generated which contains class label of each tweet in a separate line.  