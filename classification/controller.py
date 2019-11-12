from classification.utilities import *


def run_ml():
    train_df = load_data_set_from_csv()
    train_x, valid_x, train_y, valid_y = split_data(train_df)
    train_y, valid_y = encode_label_data(train_y, valid_y)
    count_vector = create_vector_object(train_df)
    x_train_count, x_valid_count = transform_train_data(count_vector, train_x, valid_x)
    x_train_tfidf, x_valid_tfidf = create_word_tf_idf(train_df, train_x, valid_x)
    x_train_tfidf_ngram, x_valid_tfidf_ngram = create_ngram_tf_idf(train_df, train_x, valid_x)
    x_train_tfidf_ngram_chars, x_valid_tfidf_ngram_chars = create_character_tf_idf(train_df, train_x, valid_x)
    test_nb_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def run_ml_from_model():
    train_df = load_test_data_set_from_csv()
    whole_df = read_data_frame()
    valid_y = train_df['labels']
    valid_y = encode_label_data_for_accuracy_test(valid_y)
    x_test_tfidf_ngram = create_ngram_tf_idf_for_ip(train_df, whole_df)
    # count_vector = create_vector_object(whole_df)
    # x_test_count = transform_ip_data(count_vector, train_df)
    fetch_accuracy_from_existing_model(x_test_tfidf_ngram, valid_y)


def n_gram_predict_from_ip(batch):
    description_list = list()
    article_ids = list()
    for article_id, description in batch:
        description_list.append(description)
        article_ids.append(article_id)
    whole_df = read_data_frame()
    plain_df = create_data_frame_for_ip(description_list)
    df = remove_stop_words(plain_df)
    x_test_tfidf_ngram = create_ngram_tf_idf_for_ip(df, whole_df)
    predictions = fetch_predictions_n_gram(x_test_tfidf_ngram)
    predicted_categories = decode_category(predictions)
    print(predictions)
    print(predicted_categories)
    predicted_dict = dict(zip(article_ids, predicted_categories))
    return predicted_dict


def count_vector_predict_from_ip(batch):
    description_list = list()
    article_ids = list()
    for article_id, description in batch:
        description_list.append(description)
        article_ids.append(article_id)
    whole_df = read_data_frame()
    plain_df = create_data_frame_for_ip(description_list)
    df = remove_stop_words(plain_df)
    count_vector = create_vector_object(whole_df)
    x_test_count = transform_ip_data(count_vector, df)
    predictions = fetch_predictions_count_vector(x_test_count)
    predicted_categories = decode_category(predictions)
    print(predictions)
    print(predicted_categories)
    predicted_dict = dict(zip(article_ids, predicted_categories))
    return predicted_dict


def word_level_predict_from_ip(batch):
    description_list = list()
    article_ids = list()
    for article_id, description in batch:
        description_list.append(description)
        article_ids.append(article_id)
    whole_df = read_data_frame()
    plain_df = create_data_frame_for_ip(description_list)
    df = remove_stop_words(plain_df)
    x_test_tfidf = create_word_tf_idf_for_ip(df, whole_df)
    predictions = fetch_predictions_word_level(x_test_tfidf)
    predicted_categories = decode_category(predictions)
    print(predictions)
    print(predicted_categories)
    predicted_dict = dict(zip(article_ids, predicted_categories))
    return predicted_dict


def char_level_predict_from_ip(batch):
    description_list = list()
    article_ids = list()
    for article_id, description in batch:
        description_list.append(description)
        article_ids.append(article_id)
    whole_df = read_data_frame()
    plain_df = create_data_frame_for_ip(description_list)
    df = remove_stop_words(plain_df)
    x_test_tfidf_ngram_chars = create_character_tf_idf_for_ip(df, whole_df)
    predictions = fetch_predictions_char_level(x_test_tfidf_ngram_chars)
    predicted_categories = decode_category(predictions)
    print(predictions)
    print(predicted_categories)
    predicted_dict = dict(zip(article_ids, predicted_categories))
    return predicted_dict
