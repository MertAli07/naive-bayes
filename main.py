import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import math
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def lapsm(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)

def label_group(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data

def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = label_group(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors

def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = lapsm(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result

def get_word_counts(X, train_labels, vocab):
    word_counts = {"sport": {}, "business": {}, "politics": {}, "entertainment": {}, "tech": {}}
    for i in range(X.shape[0]):
        l = train_labels[i]
        for j in range(len(vocab)):
            if(vocab[j] in word_counts[l].keys()):
                word_counts[l][vocab[j]] += X[i][j]
            else:
                word_counts[l][vocab[j]] = X[i][j]
    return word_counts

def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0

if __name__ == '__main__':
    #read the csv file
    df = pd.read_csv("English Dataset.csv", encoding='cp1254')
    data = df.to_numpy()  # convert it to numpy array

    # shuffle the data
    np.random.shuffle(data)

    # train-test split
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(data[:,1], data[:,2], test_size=0.2)

    # PART 2
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    print("PART 2:")
    # apply naive bayes to unigram
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_sentences)
    vocab = vectorizer.get_feature_names_out()
    X = X.toarray()
    word_counts = get_word_counts(X, train_labels, vocab)


    labels = ["sport", "business", "politics", "entertainment", "tech"]
    n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
    pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
    print("Accuracy of unigram: ", accuracy_score(test_labels,pred))

    # apply naive bayes to bigram
    vectorizer2 = CountVectorizer(ngram_range=(2,2), analyzer='word')
    X2 = vectorizer2.fit_transform(train_sentences)
    vocab2 = vectorizer2.get_feature_names_out()
    X2 = X2.toarray()
    word_counts2 = get_word_counts(X2, train_labels, vocab2)

    pred = predict(n_label_items, vocab2, word_counts2, log_label_priors, labels, test_sentences)
    print("Accuracy of bigram: ", accuracy_score(test_labels,pred))

    # PART 3
    print("PART 3:")
    # a
    category = {"sport": [], "business": [], "politics": [], "entertainment": [], "tech": []}
    for i in range(len(train_labels)):
        category[train_labels[i]].append(train_sentences[i])

    train_narrowed = []
    labels_narrowed = []
    for cat in category.keys():
        tfidfvectorizer = TfidfVectorizer(analyzer='word')
        tfidf_wm = tfidfvectorizer.fit_transform(category[cat])
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()

        matrix = tfidf_wm.toarray()
        all_texts = []
        all_words = []
        for i in matrix:
            max_ind = np.argpartition(i, -20)[-20:]
            all_texts.extend(i[max_ind])
            all_words.extend(tfidf_tokens[max_ind])

        ind = np.argpartition(all_texts, -20)[-20:]
        best10 = {}
        for i in range(len(np.array(all_words)[ind])):
            if(np.array(all_words)[ind][i] not in best10.keys()):
                best10[np.array(all_words)[ind][i]] = np.array(all_texts)[ind][i]
        best10 = {k: v for k, v in sorted(best10.items(), reverse=True ,key=lambda item: item[1])}
        best10 = {k: best10[k] for k in list(best10)[:10]}

        df = pd.DataFrame.from_dict(best10, orient="index")
        print(df)

        min_texts = []
        min_words= []

        mat = np.copy(matrix)
        mat[mat == 0] = 999
        for i in mat:
            min_ind = np.argpartition(i, 20)[:20]
            min_texts.extend(i[min_ind])
            min_words.extend(tfidf_tokens[min_ind])

        ind_min = np.argpartition(min_texts, 20)[:20]
        bottom10 = {}
        for i in range(len(np.array(min_words)[ind_min])):
            if(np.array(min_words)[ind_min][i] not in bottom10.keys()):
                bottom10[np.array(min_words)[ind_min][i]] = np.array(min_texts)[ind_min][i]
        bottom10 = {k: v for k, v in sorted(bottom10.items(), key=lambda item: item[1])}
        bottom10 = {k: bottom10[k] for k in list(bottom10)[:10]}
        dfmin = pd.DataFrame.from_dict(bottom10, orient="index")
        print(dfmin)

        train_narrowed.extend(all_words)
        temp= (cat+" ")*len(all_words)
        t_list = temp.split(" ")
        labels_narrowed.extend(t_list[:-1])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_narrowed)
    vocab = vectorizer.get_feature_names_out()
    X = X.toarray()
    word_counts = get_word_counts(X, labels_narrowed, vocab)

    n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
    pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
    print("Accuracy of best10: ", accuracy_score(test_labels,pred))

    #b
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = vectorizer.fit_transform(train_narrowed)
    vocab = vectorizer.get_feature_names_out()
    X = X.toarray()
    word_counts = get_word_counts(X, labels_narrowed, vocab)

    n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
    pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
    print("Accuracy of non-stop_word best10: ", accuracy_score(test_labels,pred))
