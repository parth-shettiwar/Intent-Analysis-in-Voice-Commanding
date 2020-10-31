This folder contains Logistic Regression applied on the dataset with various preprocessing techniques.


==> The best results are given by Logistic Regression tf file.

=> Its weights and bias has been copied and a code is written in java so no dependency has to be downloaded while running in android.

The java code can be found in ../LR in java using weights

The tflite file in this folder is created by the Logistic Refression tf file.

 
Model Explaination :-

The linear equation res = wx+b is used to train the model (w= weights and b=bias).

w is a matrix of weights where w(i,j) represents importance of word j on intent i.

x is a vector containing count of words in input string matching the vocab.

b is a vecor containing biases for each intent.

After the model is trained we get the matrix of w and b which is then copied in JAVA code.

The intent containing maximum res (wx+b) value is printed.