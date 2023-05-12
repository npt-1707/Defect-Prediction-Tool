import argparse, os, time
from urllib.parse import urlparse

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-commit', type=str, default='', help='commit link')
    parser.add_argument('-feature', type=str, default='', help='.csv file path')

    parser.add_argument('-ensemble', action='store_true', help='enable ensemble')

    available_deep_models = ['deepjit', 'cc2vec', 'simcom']
    available_traditional_models = ['lapredict', 'earl', 'tler', 'jitline']
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('-deep', nargs='+', type=str, choices=available_deep_models, help='list of deep learning models')
    model.add_argument('-traditional', nargs='+', type=str, choices=available_traditional_models, help='list of machine learning models')

    available_metrics = ['auc', 'f1']
    parser.add_argument('-metric', nargs='+', type=str, default=['auc'], choices=available_metrics, help='list of metrics')

    parser.add_argument('-debug', action='store_true', help='enable ensemble')

    return parser

# sourcery skip: raise-specific-error
if __name__ == '__main__':
    params = read_args().parse_args()

    if params.debug:
        print(params.commit)
        print(params.feature)
        print(params.ensemble)
        print(params.deep)
        print(params.traditional)
        print(params.metric)
    
    if params.feature == '' and params.commit == '':
        raise Exception("-commit or -feature is required")

    if params.feature != '':
        # Check if file exists
        if not os.path.isfile(params.feature):
            print(f'Error: {params.feature} does not exist')
        else:
            # Read file contents
            with open(params.feature, 'r') as f:
                file_contents = f.read()
                if params.debug:
                    print(file_contents)

    if params.commit != '':
        # Parse the URL and check if the hostname is github.com and the path contains /commit/
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            if params.debug:
                print(f'{params.commit} is a GitHub commit link')
        else:
            raise Exception(f'{params.commit} not a GitHub commit link')