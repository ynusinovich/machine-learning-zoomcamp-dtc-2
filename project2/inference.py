import os, json, requests
import argparse
import numpy as np
import cv2 as cv
import tensorflow as tf

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

INPUT_SIZE = 244

CLASSES = 2


def format_image_for_inference(img):
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / INPUT_SIZE
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    return new_image


def img_load(filename):
    X = []

    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    img = format_image_for_inference(img)
    img = img.astype(float) / 255.

    X.append(img)

    X = np.array(X)

    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, ))

    return result


def tune_inf_ds(dataset):
    dataset = dataset.batch(1)
    return dataset


def predict(image_url_json):
    
    saved_model_path = "model/2023_12_17_03_46_21"
    model = tf.keras.models.load_model(saved_model_path)

    image_url = image_url_json["url"]
    destination_folder = "tmp"
    os.makedirs(destination_folder, exist_ok=True)
    filename = os.path.join(destination_folder, os.path.basename(image_url))
    response = requests.get(image_url)
    with open(filename, "wb") as f:
        f.write(response.content)

    raw_inf_ds = img_load(filename)

    inf_ds = tune_inf_ds(raw_inf_ds)

    inf_list = list(inf_ds.take(1).as_numpy_iterator())

    image = inf_list[0]

    predictions = model(image)

    predicted_box = predictions[1][0] * INPUT_SIZE
    predicted_box = tf.cast(predicted_box, tf.int32)

    predicted_box = list(predicted_box.numpy())

    predicted_label = predictions[0][0]

    if predicted_label[0] > predicted_label[1]:
        label_name = "masked"
    else:
        label_name = "unmasked"

    print(label_name)
    print(predicted_box)

    result = {
        'mask': label_name,
        "box": predicted_box
    }

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get image URL.")
    parser.add_argument("--url", type=str, help="URL of image of person/people to check for masks")

    args = parser.parse_args()

    if args.url:
        try:
            image_url_json = json.loads(args.url)
            print("Image URL received")
            print(image_url_json)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON data. {e}")
    else:
        print("No URL provided. Use the --url flag to provide URL information.")
    
    result = predict(image_url_json)
    print(result)
