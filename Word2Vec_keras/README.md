This is a  Average word2vec model for text classification among 26 Intents.

Run the following commands:
pip install tensorflow
pip install keras
pip install gin-config
pip install tensorflow-datasets
pip install tensorflow-addons

Now in root directory run:
python train_classification.py

The final.png is the tsne plot of the third layer of the model(Averages the embeddings of word of sentences).

The overall model is :
1)Embed the words of input sentence to a different space of dimension 16.
2)Average the resultant vectors of words, of a sentence.
3)This is followed by Dense layers and in between a dropout layer.

The temp.txt contains the output of third layer from model.
labelstsne.txt contains the labels of input sentences.
The above txt files are used for tsne plot.

The train folder contains 26 subfolders contining all examples of 26 distinct intents.

vocab.txt contains the vocabulary generated from the dataset.

model.tflite is the tflite converted model which can be directly imported in Android and IOS apps for inference.




