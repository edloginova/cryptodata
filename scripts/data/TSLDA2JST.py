from argparse import ArgumentParser

import pandas as pd

nlp_folder = 'nlp/'

# python TSLDA2JST.py -f ../data/news/TSLDA_categories.dat -o ../data/news/JST_categories.dat
# python TSLDA2JST.py -f ../data/news/TSLDA_tokens.dat -o ../data/news/JST_tokens.dat
# python TSLDA2JST.py -f ../data/reddit/long/TSLDA_categories.dat -o ../data/reddit/long/JST_categories.dat
# python TSLDA2JST.py -f ../data/reddit/long/TSLDA_tokens.dat -o ../data/reddit/long/JST_tokens.dat
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="data_folder",
                        help="Data folder", default='')
    parser.add_argument("-f", "--filename", dest="input_filename",
                        help="Input filename", default='../../JST_debug/JST_tokens.dat')
    parser.add_argument("-o", "--output", dest="output_filename",
                        help="Output filename", default='../../JST_debug/JST_new_tokens.dat')
    args = parser.parse_args()
    args = vars(args)
    with open(args['input_filename'], 'r', encoding='utf-8') as f:
        content = f.readlines()
    data = {}
    for line in content:
        doc_idx, sent_idx, *tokens = line.split()
        data[len(data)] = {'doc_idx': doc_idx, 'tokens': tokens}
    df = pd.DataFrame.from_dict(data, orient='index')
    dt = df.groupby('doc_idx')['tokens'].apply(list)
    dt = pd.DataFrame(dt)
    dt['index'] = [int(x[1:]) for x in dt.index]
    dt = dt.set_index('index')
    dt = dt.sort_index()
    dt['tokens'] = dt['tokens'].apply(lambda x: [y for z in x for y in z])
    dt['tokens'] = dt['tokens'].apply(lambda x: ' '.join(x))
    with open(args['output_filename'], 'w', encoding='utf-8') as f:
        for idx, row in dt.iterrows():
            f.write('d' + str(idx) + ' ' + row['tokens'] + '\n')
