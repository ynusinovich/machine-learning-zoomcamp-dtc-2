import json
from flask import Flask
from flask import request
from flask import jsonify

from inference import predict

app = Flask('mask')


@app.route('/inference_app', methods=['POST'])
def inference_app():
    mask = request.get_json()

    result = predict(mask)

    result["box"] = [float(i) for i in result["box"]]

    json_result = json.dumps(result)

    return jsonify(result=json_result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)