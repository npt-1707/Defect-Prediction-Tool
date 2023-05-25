import argparse, os, time, requests
from urllib.parse import urlparse

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-commit', type=str, default='', help='commit link')
    parser.add_argument('-access_token', type=str, default='', help='user access token')
    parser.add_argument('-feature', type=str, default='', help='.csv file path')

    parser.add_argument('-ensemble', action='store_true', help='enable ensemble')

    available_deep_models = ['deepjit', 'cc2vec', 'simcom', 'codebert_cc2vec']
    available_traditional_models = ['lapredict', 'earl', 'tler', 'jitline']
    model = parser.add_mutually_exclusive_group()
    model.add_argument('-deep', nargs='+', type=str, default=[], choices=available_deep_models, help='list of deep learning models')
    model.add_argument('-traditional', nargs='+', type=str, default=[], choices=available_traditional_models, help='list of machine learning models')

    parser.add_argument('-debug', action='store_true', help='allow debug print')

    return parser

# sourcery skip: raise-specific-error
if __name__ == '__main__':
    params = read_args().parse_args()

    if params.debug:
        print("Commit link: ", params.commit)
        print("Access token: ", params.access_token)
        print("Feature:", params.feature)
        print("Ensemble:", params.ensemble)
        print("DL models:", params.deep)
        print("ML models:", params.traditional)

    request = {
        "id": "main-" + str(int(time.time())),
        'ensemble': params.ensemble,
        "deep_models": params.deep,
        "traditional_models": params.traditional,
        "number_models": len(params.deep) + len(params.traditional)
    }
    
    if params.feature == '' and params.commit == '':
        raise Exception("-commit or -feature is required")

    if params.feature != '':
        if len(params.traditional) == 0:
            raise Exception(f'Atleast 1 traditional model is required')
        # Check if file exists
        if not os.path.isfile(params.feature):
            raise Exception(f'{params.feature} does not exist')
        else:
            # Read file contents
            with open(params.feature, 'r') as f:
                file_contents = f.read()
                if params.debug:
                    print(file_contents)
                request['features'] = file_contents

    if params.commit != '':
        if len(params.deep) == 0:
            raise Exception(f'Atleast 1 deep learning model is required')
        if params.access_token == '':
            raise Exception(f'Github access token is required')
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            if params.debug:
                print(f'{params.commit} is a GitHub commit link')
            request['link_commit'] = params.commit
            request['access_token'] = params.access_token
        else:
            raise Exception(f'{params.commit} not a GitHub commit link')
        
    if params.debug:
        print(request)

    response = requests.post('http://localhost:5000/api/input_output', json=request)
    if response.status_code == 200:
        print(response.json())  # {'message': 'accepted'}
    else:
        raise Exception(response.status_code)