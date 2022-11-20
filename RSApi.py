from flask import Flask, request, jsonify, make_response, abort
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import keras
import json
import heapq

app = Flask(__name__)
app.config["DEBUG"] = True

reconstructed_model = keras.models.load_model(
        "./NeuMF_8_[64, 32, 16, 8]_1668402292.h5")

@app.route('/api/v1/resources/books', methods=['POST'])
def create_task():
    if not request.json :
        abort(400)
    bookIds = request.json["bookIds"]
    userId = request.json["userId"]
    userIds = np.full(len(bookIds), userId, dtype="int32");
    results = reconstructed_model.predict([tf.convert_to_tensor(userIds, dtype=tf.int32), np.array(tf.convert_to_tensor(bookIds, dtype=tf.int32))],
                                batch_size=100, verbose=0)
    map_item_score = {}
    for i in range(len(bookIds)):
          item = bookIds[i]
          map_item_score[item] = results[i]
    bookIds.pop()
      # Evaluate top rank list
    ranklist = np.array(heapq.nlargest(10, map_item_score, key=map_item_score.get))
    results = ranklist.flatten()
    json_str = json.dumps(results.tolist())
    print(json_str)
    
    return jsonify(json_str), 201