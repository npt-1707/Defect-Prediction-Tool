from flask import Flask
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/crawl')
def get_commits():
    url = "https://github.com/manhlamabc123/Tic-tac-toe-Game-using-Network-Socket-APIs/commit/5f2188e336add1b6798c822c0c180d2603d75807"
    headers = {'Accept': 'application/vnd.github+json'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        commits = response.json()
        return commits
    else:
        return None
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")