import argparse

def read_args():
    parser = argparse.ArgumentParser()

    model_input = parser.add_mutually_exclusive_group(required=True)
    model_input.add_argument('-commit', type=str, default='', help='commit link')
    model_input.add_argument('-feature', type=str, default='', help='.csv file path')

    parser.add_argument('-ensemble', action='store_true', help='enable ensemble')

    available_deep_models = ['deepjit', 'cc2vec', 'simcom']
    available_traditional_models = ['lapredict', 'earl', 'tler', 'jitline']
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('-deep', nargs='+', type=str, choices=available_deep_models, help='list of deep learning models')
    model.add_argument('-traditional', nargs='+', type=str, choices=available_traditional_models, help='list of machine learning models')

    parser.add_argument('-metric', nargs='+', type=str, default=['auc'], help='list of metrics')

    parser.add_argument('-debug', action='store_true', help='enable ensemble')

    return parser

if __name__ == '__main__':
    params = read_args().parse_args()

    if params.debug:
        print(params.commit)
        print(params.feature)
        print(params.ensemble)
        print(params.deep)
        print(params.traditional)
        print(params.metric)
    
    