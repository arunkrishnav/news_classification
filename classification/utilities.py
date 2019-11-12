import numpy
import string
import textblob
import pickle
import time
import json
import re

from pandas import DataFrame, read_csv, read_pickle
from sklearn import model_selection, preprocessing, decomposition, metrics, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


def load_stop_words_from_csv():
    stop_words = read_csv('classification/data/stop_words.csv', names=["stop_words"])
    return stop_words["stop_words"].values


def remove_stop_words(df):
    stop_words = load_stop_words_from_csv()
    punch = str.maketrans("", "", string.punctuation)
    df["text"] = df['text'].apply(lambda x: ' '.join([word for word in str(re.sub('\s+', " ", str(x).translate(punch)))
                                                     .split() if word not in stop_words and not (word.isalpha()
                                                                                                 or word.isalnum())]))
    return df


def load_data_set_from_csv():
    """ load the data set """
    time_stamp = time.time()
    names = ["labels", "text"]
    plain_df = read_csv('classification/data/train_data.csv', names=names)
    train_df = remove_stop_words(plain_df)
    train_df.to_pickle('classification/data_frame/df_%s.pkl' % time_stamp)
    return train_df


def load_test_data_set_from_csv():
    """ load the data set """
    # time_stamp = time.time()
    names = ["labels", "text"]
    plain_df = read_csv('classification/data/test_data.csv', names=names)
    train_df = remove_stop_words(plain_df)
    # train_df.to_pickle('classification/data_frame/df_%s.pkl' % time_stamp)
    return train_df


def read_data_frame():
    df = read_pickle('classification/data_frame/df.pkl')
    return df


def create_data_frame(labels, texts):
    """ create a data frame using texts and labels """
    train_df = DataFrame()
    train_df["text"] = texts
    train_df["labels"] = labels
    return train_df


def create_data_frame_for_ip(text):
    """ create a data frame using texts and labels """
    train_df = DataFrame()
    train_df["text"] = text
    return train_df


def split_data(train_df):
    """ split the data set into training and validation data sets """
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], train_df['labels'],
                                                                          test_size=0.001)
    return train_x, valid_x, train_y, valid_y


def encode_label_data(train_y, valid_y):
    """ label encode the target variable """
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    encoded_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    categories = encoder.inverse_transform(encoded_labels)
    data = dict(zip(encoded_labels, categories))
    valid_y = encoder.fit_transform(valid_y)
    time_stamp = time.time()
    with open('classification/data_frame/mapping_%s.json' % time_stamp, 'w') as outfile:
        json.dump(data, outfile)
    return train_y, valid_y


def encode_label_data_for_accuracy_test(valid_y):
    """ label encode the target variable """
    encoder = preprocessing.LabelEncoder()
    valid_y = encoder.fit_transform(valid_y)
    return valid_y


def create_vector_object(train_df):
    """ create a count vector object """
    count_vector = CountVectorizer(analyzer=lambda x: x.split(), token_pattern=r'\w{1,}', max_features=9000,
                                   ngram_range=(2, 3))
    count_vector.fit(train_df['text'])
    return count_vector


def transform_train_data(count_vector, train_x, valid_x):
    """ transform the training and validation data using count vector object """
    x_train_count = count_vector.transform(train_x)
    x_valid_count = count_vector.transform(valid_x)
    return x_train_count, x_valid_count


def transform_ip_data(count_vector, df):
    """ transform the training and validation data using count vector object """
    x_test_count = count_vector.transform(df['text'])
    return x_test_count


def create_word_tf_idf(train_df, train_x, valid_x):
    """ word level tf-idf """
    tfidf_vector = TfidfVectorizer(analyzer='word', max_features=5000)
    tfidf_vector.fit(train_df['text'])
    x_train_tfidf = tfidf_vector.transform(train_x)
    x_valid_tfidf = tfidf_vector.transform(valid_x)
    return x_train_tfidf, x_valid_tfidf


def create_word_tf_idf_for_ip(df, whole_df):
    """ word level tf-idf """
    tfidf_vector = TfidfVectorizer(analyzer='word', max_features=5000)
    tfidf_vector.fit(whole_df['text'])
    x_test_tfidf = tfidf_vector.transform(df['text'])
    return x_test_tfidf


def create_ngram_tf_idf(train_df, train_x, valid_x):
    """ ngram level tf-idf """
    tfidf_vector_ngram = TfidfVectorizer(analyzer=lambda x: x.split(), ngram_range=(1, 2), max_features=7000)
    tfidf_vector_ngram.fit(train_df['text'])
    x_train_tfidf_ngram = tfidf_vector_ngram.transform(train_x)
    x_valid_tfidf_ngram = tfidf_vector_ngram.transform(valid_x)
    return x_train_tfidf_ngram, x_valid_tfidf_ngram


def create_ngram_tf_idf_for_ip(df, whole_df):
    """ ngram level tf-idf """
    tfidf_vector_ngram = TfidfVectorizer(analyzer=lambda x: x.split(), ngram_range=(1, 2), max_features=7000)
    tfidf_vector_ngram.fit(whole_df['text'])
    x_test_tfidf_ngram = tfidf_vector_ngram.transform(df['text'])
    return x_test_tfidf_ngram


def create_character_tf_idf(train_df, train_x, valid_x):
    """ characters level tf-idf """
    tfidf_vector_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                               max_features=5000)
    tfidf_vector_ngram_chars.fit(train_df['text'])
    x_train_tfidf_ngram_chars = tfidf_vector_ngram_chars.transform(train_x)
    x_valid_tfidf_ngram_chars = tfidf_vector_ngram_chars.transform(valid_x)
    return x_train_tfidf_ngram_chars, x_valid_tfidf_ngram_chars


def create_character_tf_idf_for_ip(df, whole_df):
    """ characters level tf-idf """
    tfidf_vector_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                               max_features=5000)
    tfidf_vector_ngram_chars.fit(whole_df['text'])
    x_test_tfidf_ngram_chars = tfidf_vector_ngram_chars.transform(df['text'])
    return x_test_tfidf_ngram_chars


def load_pre_trained_embedded_vector():
    """ load the pre-trained word-embedding vectors """
    embeddings_index = {}
    for i, line in enumerate(open('classification/data/wiki.ml.vec', encoding='utf-8')):
        try:
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
    return embeddings_index


def create_tokenizer(train_df):
    """ create a tokenizer """
    token = text.Tokenizer()
    token.fit_on_texts(train_df['text'])
    word_index = token.word_index
    return token, word_index


def convert_text_to_token(train_x, valid_x, token):
    """ convert text to sequence of tokens and pad them to ensure equal length vectors """
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
    return train_seq_x, valid_seq_x


def map_token_embedding(word_index, embeddings_index):
    """ create token-embedding mapping """
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def character_count(train_df):
    """ total number of characters in the documents """
    train_df['char_count'] = train_df['text'].apply(len)
    return train_df


def word_count(train_df):
    """ total number of words in the documents """
    train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
    return train_df


def word_density(train_df):
    """ average length of the words used in the documents """
    train_df['word_density'] = train_df['char_count'] / (train_df['word_count'] + 1)
    return train_df


def punctuation_count(train_df):
    """ total number of punctuation marks in the documents """
    train_df['punctuation_count'] = train_df['text'].apply(lambda x: len("".join(_ for _ in x if _ in
                                                           string.punctuation)))
    return train_df


def title_word_count(train_df):
    """ total number of proper case (title) words in the documents """
    train_df['title_word_count'] = train_df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    return train_df


def upper_case_word_count(train_df):
    """ total number of upper count words in the documents """
    train_df['upper_case_word_count'] = train_df['text'].apply(lambda x: len([wrd for wrd in x.split() if
                                                                              wrd.isupper()]))
    return train_df


def check_pos_tag(x, flag):
    """ to check and get the part of speech tag count of a words in a given sentence """
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }

    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except Exception as e:
        # TODO log write
        pass
    return cnt


def part_of_speech_frequency(train_df):
    """ frequency distribution of Part of Speech Tags """
    train_df['noun_count'] = train_df['text'].apply(lambda x: check_pos_tag(x, 'noun'))
    train_df['verb_count'] = train_df['text'].apply(lambda x: check_pos_tag(x, 'verb'))
    train_df['adj_count'] = train_df['text'].apply(lambda x: check_pos_tag(x, 'adj'))
    train_df['adv_count'] = train_df['text'].apply(lambda x: check_pos_tag(x, 'adv'))
    train_df['pron_count'] = train_df['text'].apply(lambda x: check_pos_tag(x, 'pron'))
    return train_df


def train_lda_model(x_train_count, count_vector):
    """ train a LDA Model """
    lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
    x_topics = lda_model.fit_transform(x_train_count)
    topic_word = lda_model.components_
    vocab = count_vector.get_feature_names()
    return x_topics, topic_word, vocab


def view_model(topic_word, vocab):
    """ view the topic models """
    n_top_words = 10
    topic_summaries = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_summaries.append(' '.join(topic_words))
    return topic_summaries


def create_cnn(word_index, embedding_matrix):
    """ creating a Convolutional neural network"""

    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, name, is_neural_net=False):
    """ model training """

    # fit the training data set on the classifier
    classifier.fit(feature_vector_train, label)

    # write classifier to a file
    time_stamp = time.time()
    filename = 'classification/classified_model/finalized_model_%s_%s.sav' % (time_stamp, name)
    pickle.dump(classifier, open(filename, 'wb'))

    # predict the labels on validation data set
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


def test_cnn_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y):
    classifier = create_cnn(word_index, embedding_matrix)
    accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True, name='cnn')
    print("CNN, Word Embeddings", accuracy)


def test_nb_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("NB, WordLevel TF-IDF: ", accuracy)

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("NB, N-Gram Vectors: ", accuracy)

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("NB, CharLevel Vectors: ", accuracy)


def fetch_accuracy_from_existing_model(feature_vector_valid, valid_y):
    """return accuracy from exiting model"""

    loaded_model = pickle.load(open('classification/classified_model/nb_n_gram_vector.sav',
                                    'rb'))
    predictions = loaded_model.predict(feature_vector_valid)
    accuracy = metrics.accuracy_score(predictions, valid_y)
    print("NB, n-gram: ", accuracy)


def fetch_predictions_n_gram(feature_vector_valid):
    loaded_model = pickle.load(open('classification/classified_model/nb_n_gram_vector.sav',
                                    'rb'))
    predictions = loaded_model.predict(feature_vector_valid)
    return predictions


def fetch_predictions_count_vector(feature_vector_valid):
    loaded_model = pickle.load(open('classification/classified_model/nb_count_vector.sav',
                                    'rb'))
    predictions = loaded_model.predict(feature_vector_valid)
    return predictions


def fetch_predictions_word_level(feature_vector_valid):
    loaded_model = pickle.load(open('classification/classified_model/nb_word_level.sav',
                                    'rb'))
    predictions = loaded_model.predict(feature_vector_valid)
    return predictions


def fetch_predictions_char_level(feature_vector_valid):
    loaded_model = pickle.load(open('classification/classified_model/nb_char_level_vector.sav', 'rb'))
    predictions = loaded_model.predict(feature_vector_valid)
    return predictions


def decode_category(encoded_array):
    with open('classification/data_frame/mapping_1532324929.800722.json') as f:
        data = json.load(f)
    category_list = list()
    for label in encoded_array:
        category_list.append(data[str(label)])
    return category_list
