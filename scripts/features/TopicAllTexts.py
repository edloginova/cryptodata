from TopicExtractor import *

verbose = True
nlp_folder = '../../models/topic/'
DATA_FOLDER = '../../data/processed/'
datasets = {'news': DATA_FOLDER + 'news/unique_comments.p',
            'reddit_long': DATA_FOLDER + 'reddit/long/unique_comments.p'}
all_text_data = []
for model_name, input_filename in datasets.items():
    print(model_name, '---', input_filename)
    _, text_data = load_data(input_filename, verbose)
    all_text_data.extend(text_data)

len(all_text_data)
save_intermediate = True
num_topics = 5
model_name = 'all-texts'
ldamodel, dictionary = train_lda(all_text_data, verbose, save_intermediate, num_topics, model_name)
