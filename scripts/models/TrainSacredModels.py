import os
import subprocess
import sys
from argparse import ArgumentParser

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions import determine_data_source, slack_message

__author__ = "Ekaterina Loginova"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"
DATA_FOLDER = '../../data/processed/'
SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
SLACK_USER = 'ekaterina.loginova'


def formulate_model_name(args, model_type, text_features):
    result = "\""
    if args['model_name'] != '':
        result += args['model_name'] + '_'
    name = '_'.join(
        [determine_data_source(args['basetable_filename']), model_type,
         args['tuning_mode'],
         str(args['standardize']), text_features, args['coin_name'], str(int(args['next_day']))])
    result += name + "\""
    return result


# python TrainSacredModels.py -d ../../data/processed/news/ -f new_basetable_noNAN.p -a 1
# python TrainSacredModels.py  -m svm -af 1
if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="input_dir",
                           help="The path to the directory with input files.", default='')
    argparser.add_argument("-f", "--filename", dest="basetable_filename",
                           help="Basetable filename", default="basetable_nonan.p")
    argparser.add_argument("-m", "--model", dest="model_type",
                           help="Machine learning model type (svm / nb / logreg / rf / xgboost)", default='xgboost')
    argparser.add_argument("-t", "--tuning", dest="tuning_mode",
                           help="Cross-validation optimization mode (grid / rand)", default='rand')
    argparser.add_argument("-st", "--standardize", dest="standardize", type=bool,
                           help="Standardize or not", default=True)
    argparser.add_argument("-p", "--prop", dest="train_test_prop", type=float,
                           help="Train/test dataset size proportion (default: 0.8)", default=0.8)
    argparser.add_argument("-tf", "--text_features", dest="text_features",
                           help="Which text features to use (none / all / polarity / topic / subjectivity). Can combine via + (e.g., polarity+topic). if not specified, only length of texts is used.")
    argparser.add_argument("-tr", "--timerange", dest="time_range",
                           help="Time range string in the following format: 31/12/2019-31/12/2020.")
    argparser.add_argument("-s", "--score", dest="score_metric",
                           help="Score metric to guide grid search (roc_auc / accuracy / f1)", default='roc_auc')
    argparser.add_argument("-sv", "--save", dest="save", type=bool,
                           help="Save the model file", default=True)
    argparser.add_argument("-a", "--all", dest="all_comb",
                           help="Run all combinations of models and text features.", action='store_true')
    argparser.add_argument("-af", "--all_files", dest="all_files_comb",
                           help="Run model on all combinations of files and text features.", action='store_true')
    argparser.add_argument("-am", "--all_models", dest="all_models_comb",
                           help="Run all models on this combinations of text features.", action='store_true')
    argparser.add_argument("-n", "--model_name", dest="model_name",
                           help="Model name for saving the files.", default='')
    argparser.add_argument("-ml", "--max_lag", dest="max_lag", default=7, type=int)
    argparser.add_argument("-c", "--coin_name", dest="coin_name", default="BTC")
    argparser.add_argument("-se", "--seed", dest="seed", default=123, type=int)
    argparser.add_argument("-sa", "--sacred", dest="sacred", default='sacred_08032021',
                           help="Sacred Mongo database (for sacredboard).")
    argparser.add_argument("-nd", "--next_day", dest="next_day", action='store_true',
                           help='Whether to predict next day (instead of the current).')
    argparser.add_argument("-su", "--slack_user", dest="slack_user", default='ekaterina.loginova')

    args = argparser.parse_args()
    args = vars(args)
    SLACK_USER = args['slack_user']
    if args['input_dir'][-1] != '/':
        args['input_dir'] = args['input_dir'] + '/'
    if args['input_dir'] not in args['basetable_filename']:
        args['basetable_filename'] = args['input_dir'] + args['basetable_filename']

    if args['all_comb']:
        for model_type in ['neural_lstm', 'nb', 'logreg', 'rf', 'svm', 'neural_mlp', 'xgboost']:
            for text_features in ['none', 'only', 'all', 'polarity', 'topic', 'subjectivity']:
                train_model_call = os.path.dirname(os.path.abspath(__file__)) + "//TrainSacredModel.py"
                try:
                    subprocess.check_output(
                        ['python', train_model_call, '-m', args['sacred'], '--name',
                         formulate_model_name(args, model_type, text_features), 'with',
                         'model_type=' + model_type, 'text_features=' + text_features,
                         'basetable_filename=' + args['basetable_filename'],
                         "max_lag=" + str(args['max_lag']),
                         "coin_name=" + args['coin_name'], "seed=" + str(args['seed']),
                         "model_name=" + formulate_model_name(args, model_type, text_features)],
                        stderr=subprocess.STDOUT, shell=False)
                except subprocess.CalledProcessError as e:
                    # slack_message('Error {} during training models. (Basetable: {})'.format(e, args['basetable_filename']),
                    #               SLACK_USER)
                    raise RuntimeError(
                        "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        slack_message('Finished training models. (Basetable: {})'.format(args['basetable_filename']), SLACK_USER)
    elif args['all_files_comb']:
        for basetable_filename in [DATA_FOLDER + 'bitcointalk/new_basetable_nonan.p',
                                   DATA_FOLDER + 'news/new_basetable_nonan.p',
                                   DATA_FOLDER + 'reddit/new_basetable_nonan.p']:
            for text_features in ['none', 'only', 'all', 'polarity', 'topic', 'subjectivity']:
                train_model_call = os.path.dirname(os.path.abspath(__file__)) + "//TrainSacredModel.py"
                try:
                    subprocess.check_output(
                        ['python', train_model_call, '-m', args['sacred'], '--name',
                         formulate_model_name(args, args['model_type'], text_features), 'with',
                         'model_type=' + args['model_type'], 'text_features=' + text_features,
                         'basetable_filename=' + basetable_filename,
                         "max_lag=" + str(args['max_lag']),
                         "coin_name=" + args['coin_name'], "next_day=" + args['next_day'], "seed=" + str(args['seed'])],
                        stderr=subprocess.STDOUT,
                        shell=False)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    elif args['all_models_comb']:
        for model_type in ['neural_lstm', 'nb', 'logreg', 'rf', 'svm', 'neural_mlp', 'xgboost']:
            train_model_call = os.path.dirname(os.path.abspath(__file__)) + "//TrainSacredModel.py"

            try:

                subprocess.check_output(
                    ['python', train_model_call, '-m', args['sacred'], '--name',
                     formulate_model_name(args, model_type, args['text_features']), 'with',
                     'model_type=' + model_type, 'text_features=' + args['text_features'],
                     'basetable_filename=' + args['basetable_filename'],
                     "max_lag=" + str(args['max_lag']),
                     "coin_name=" + args['coin_name'], "seed=" + str(args['seed']),
                     "model_name=" + formulate_model_name(args, model_type, args['text_features'])],
                    stderr=subprocess.STDOUT, shell=False)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    else:
        train_model_call = os.path.dirname(os.path.abspath(__file__)) + "//TrainSacredModel.py"
        try:
            subprocess.check_output(
                ['python', train_model_call, '-m', args['sacred'], '--name',
                 formulate_model_name(args, args['model_type'], args['text_features']), 'with',
                 'model_type=' + args['model_type'], 'text_features=' + args['text_features'],
                 'basetable_filename=' + args['basetable_filename'],
                 "max_lag=" + str(args['max_lag']),
                 "coin_name=" + args['coin_name'], "seed=" + str(args['seed']),
                 "model_name=" + formulate_model_name(args, args['model_type'], args['text_features'])],
                stderr=subprocess.STDOUT, shell=False)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))