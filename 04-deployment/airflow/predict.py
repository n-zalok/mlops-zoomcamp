import mlflow
from flask import Flask, request, jsonify


mlflow.set_tracking_uri("http://localhost:5000")
print("Connected to MLflow tracking server.")

run_id = "07d65328ad174c4e85103ba16ff0299a"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/models")
print("Model loaded successfully.")


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': run_id,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)