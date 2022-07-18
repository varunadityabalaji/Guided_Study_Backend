from flask import Flask
from flask_cors import CORS
from flask_cors import CORS, cross_origin
from flask import jsonify
import json
from flask import Flask, request
import predict
import numpy as np

app = Flask(__name__)
Cors = CORS(app)
CORS(app, resources={r'/*': {'origins': '*'}},CORS_SUPPORTS_CREDENTIALS = True)
app.config['CORS_HEADERS'] = 'Content-Type'




@app.route("/predict", methods=["POST"])
def index():
    record = json.loads(request.data)
    print(record['ticker'])
    # a = np.array([1,2,3,4,5]).tolist()
    response = predict.predict(record['ticker'])
    return json.dumps(response)
    # return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)