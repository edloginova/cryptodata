import subprocess
from argparse import ArgumentParser


def run_python_script(command):
    try:
        subprocess.check_output(command.split(), stderr=subprocess.STDOUT, shell=False)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="input_dir",
                           help="The path to the directory with input files.", default='')
    argparser.add_argument("-f", "--filename", dest="texts_filename",
                           help="Texts filename", default="unique_comments.p")
    argparser.add_argument("-tn", "--topic_num", dest="topic_num",
                           help="Number of topics for LDA", type=int, default=5)
    argparser.add_argument("-nd", "--next_day", dest="next_day",
                           help="Next day prediction or not", action='store_false')
    args = argparser.parse_args()
    args = vars(args)
    input_dir = args['input_dir']
    texts_filename = args['texts_filename']
    topic_num = args['topic_num']
    next_day = args['next_day']
    # subprocess.run(["cd", os.getcwd() + "data/"], shell=True)
    # print(os.getcwd())
    # command = 'python data/TextCleaner.py -d {} -f {}'.format(input_dir, texts_filename)
    # run_python_script(command)
    # print('Cleaned texts.')
    # command = 'python data/ProbModelFormatter.py -d {} -f {}'.format(input_dir, texts_filename)
    # run_python_script(command)
    # print('Formatted for ABSA.')
    # # subprocess.run(["cd", "../features/"], shell=True)
    # command = 'python features/TopicExtractor.py -d {} -f {} -n {}'.format(input_dir, texts_filename, topic_num)
    # run_python_script(command)
    # print('Extracted LDA topics.')
    # command = 'python features/TextFeatureExtractor.py  -d {} -ts post+comment'.format(input_dir)
    # run_python_script(command)
    # print('Extracted text features.')
    # # subprocess.run(["cd", "../data/"], shell=True)
    # command = 'python data/BasetableCreator.py -d {}'.format(input_dir)
    # run_python_script(command)
    # print('Created basetable.')
    # subprocess.run(["cd", "../models/"], shell=True)
    command = 'python models/TrainSacredModels.py -d {} -a'.format(input_dir)
    if next_day:
        command += ' -nd'
    run_python_script(command)
    print('Trained models.')
