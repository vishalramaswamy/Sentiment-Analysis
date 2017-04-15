import sys
import collections
from sklearn.naive_bayes import *
from sklearn.linear_model import*
import nltk
import random
import numpy
random.seed(0)
from gensim.models.doc2vec import *

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])

def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)

def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
	
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    pos_train_dict={}
    for words in train_pos:
	for word in list(set(words)):
		if word not in stopwords:
			pos_train_dict[word]=pos_train_dict.get(word,0)+1
    neg_train_dict={}
    for words in train_neg:
	for word in list(set(words)):
		if word not in stopwords:
			neg_train_dict[word]=neg_train_dict.get(word,0)+1

    features=[]
    for word in pos_train_dict:
	if pos_train_dict[word]>=1*len(train_pos)/100 and pos_train_dict[word]>=2*neg_train_dict[word]:
		features.append(word);
    for word in neg_train_dict:
	if neg_train_dict[word]>=1*len(train_neg)/100 and neg_train_dict[word]>=2*pos_train_dict[word]:
		features.append(word);

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec=[]
    for words in train_pos:
	temp=[]	
	for word in features:
		if word in words:
			temp.append(1)
		else:
			temp.append(0)
	train_pos_vec.append(temp)

    train_neg_vec=[]
    for words in train_neg:
	temp=[]	
	for word in features:
		if word in words:
			temp.append(1)
		else:
			temp.append(0)
	train_neg_vec.append(temp)
    test_pos_vec=[]
    for words in test_pos:
	temp=[]
	for word in features:
		if word in words:
			temp.append(1)
		else:
			temp.append(0)
	test_pos_vec.append(temp)

    test_neg_vec=[]
    for words in test_neg:
	temp=[]
	for word in features:
		if word in words:
			temp.append(1)
		else:
			temp.append(0)
	test_neg_vec.append(temp)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    # used TaggedDocument instead of LabeledSentence to avoid warnings
    labeled_train_pos = []
    for i in range(0, len(train_pos)) :
        labeled_train_pos.append(TaggedDocument(train_pos[i], ['TRAIN_POS_' + str(i)]))
		
    labeled_train_neg = []
    for i in range(0, len(train_neg)) :
	labeled_train_neg.append(TaggedDocument(train_neg[i], ['TRAIN_NEG_' + str(i)]))
		
    labeled_test_pos = []
    for i in range(0, len(test_pos)) :
	labeled_test_pos.append(TaggedDocument(test_pos[i], ['TEST_POS_' + str(i)]))
	
    labeled_test_neg = []
    for i in range(0, len(test_neg)) :
	labeled_test_neg.append(TaggedDocument(test_neg[i], ['TEST_NEG_' + str(i)]))
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec=[]
    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs['TRAIN_POS_' + str(i)])
    train_neg_vec=[]
    for i in range(len(train_neg)):
        train_neg_vec.append(model.docvecs['TRAIN_NEG_' + str(i)])
    test_pos_vec=[]
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs['TEST_POS_' + str(i)])
    test_neg_vec=[]
    for i in range(len(test_neg)):
	test_neg_vec.append(model.docvecs['TEST_NEG_' + str(i)])    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB(alpha=1.0,binarize=None)
    nb_model.fit(numpy.array(train_pos_vec + train_neg_vec), numpy.array(Y))
    lr_model = LogisticRegression()
    lr_model.fit(numpy.array(train_pos_vec + train_neg_vec), numpy.array(Y))
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model.fit(numpy.array(train_pos_vec + train_neg_vec), numpy.array(Y))
    lr_model = LogisticRegression()
    lr_model.fit(numpy.array(train_pos_vec + train_neg_vec), numpy.array(Y))
   
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    tp = 0
    fn = 0
    p_predict = model.predict(test_pos_vec)
    for item in p_predict:
        if item == "pos":
            tp = tp + 1
        else:
            fn = fn + 1
    n_predict = model.predict(test_neg_vec)
    fp = 0
    tn = 0
    for item in n_predict:
        if item == "neg":
            tn = tn + 1
        else:
            fp = fp + 1            
    accuracy = float(tp + tn) / (tp + tn + fn + fp)  
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
