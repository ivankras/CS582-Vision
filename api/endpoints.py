import os
import service as service
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/getResult', methods=['POST'])
@cross_origin()
def getResult():
    """
    Get input image urls and return result of trained model
    """
    urls = request.json
    
    return send_file(service.train_and_get_result(urls), mimetype='image/png')

@app.route('/api/getDefault', methods=['POST'])
@cross_origin()
def getDefault():
    """
    Get input image urls and return result of trained model
    """
    urls = request.json
    return send_file(service.train_and_get_result(urls), mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)