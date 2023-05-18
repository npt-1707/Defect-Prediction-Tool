from flask import Flask, request

app = Flask(__name__)

@app.route('/api/model_template', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "input": input_type
        }
    '''
    request_data = request.get_json()
    if app.debug:
        print(request_data)

    # Create model and Load pretrain

    # Forward
    if app.debug:
        output = 0

    # Create response like form below:
    '''
        {
            "id": string,
            "output": dictionary
        }
    '''
    return {
        'id': request_data["id"],
        'output': output
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)