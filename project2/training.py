import os, random, json, datetime
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

INPUT_SIZE = 244

CLASSES = 2


def list_files(full_data_path = "data/obj/", image_ext = '.jpg', split_percentage = [70, 20]):

    files = []

    discarded = 0
    masked_instance = 0

    for r, d, f in os.walk(full_data_path):
        for file in f:
            if file.endswith(".txt"):

                # first, let's check if there is only one object
                with open(full_data_path + "/" + file, 'r') as fp:
                    lines = fp.readlines()
                    if len(lines) > 1:
                        discarded += 1
                        continue


                strip = file[0:len(file) - len(".txt")]
                # secondly, check if the paired image actually exist
                image_path = full_data_path + "/" + strip + image_ext
                if os.path.isfile(image_path):
                    # checking the class. '0' means masked, '1' for unmasked
                    if lines[0][0] == '0':
                        masked_instance += 1
                    files.append(strip)

    size = len(files)
    print(str(discarded) + " file(s) discarded")
    print(str(size) + " valid case(s)")
    print(str(masked_instance) + " are masked cases")

    random.shuffle(files)

    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)

    return files[0:split_training], files[split_training:split_validation], files[split_validation:]


def format_image(img, box):
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / INPUT_SIZE
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = box[0], box[1], box[2], box[3]
    new_box = [int((x - 0.5*w)* width / r), int((y - 0.5*h) * height / r), int(w*width / r), int(h*height / r)]

    return new_image, new_box


def data_load(files, full_data_path = "data/obj/", image_ext = ".jpg", train_aug = False):
    X = []
    Y = []

    for file in files:
        img = cv.imread(os.path.join(full_data_path, file + image_ext), cv.IMREAD_GRAYSCALE)

        k = 1

        with open(full_data_path + "/" + file + ".txt", 'r') as fp:
            line = fp.readlines()[0]
            if line[0] == '0':
                k = 0

            box = np.array(line[1:].split(), dtype=float)

        img, box = format_image(img, box)
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / INPUT_SIZE
        label = np.append(box, k)

        if train_aug:

            # data augmentation
            img = tf.expand_dims(img, axis=-1)  # Add channel dimension
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)

            # adjust bounding box for flips
            if tf.random.uniform(()) > 0.5:
                box = [1.0 - box[2], box[1], 1.0 - box[0], box[3]]

            if tf.random.uniform(()) > 0.5:
                box = [box[0], 1.0 - box[3], box[2], 1.0 - box[1]]

            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

            img = tf.squeeze(img, axis=-1)  # Remove channel dimension

        X.append(img)
        Y.append(label)

    X = np.array(X)

    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result


def format_instance(image, label):
    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])


def tune_training_ds(dataset, batch_size):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(len(validation_files) // 4)
    dataset = dataset.repeat()
    return dataset


def build_feature_extractor(inputs, dropout_factor):

    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(dropout_factor)(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    return x


def build_model_adapter(inputs):
  x = tf.keras.layers.Flatten()(inputs)
  x = tf.keras.layers.Dense(256, activation='relu')(x)
  return x


def build_classifier_head(inputs):
  return tf.keras.layers.Dense(CLASSES, activation='softmax', name = 'classifier_head')(inputs)


def build_regressor_head(inputs):
    return tf.keras.layers.Dense(units = '4', name = 'regressor_head')(inputs)


def build_model(inputs, dropout_factor):

    feature_extractor = build_feature_extractor(inputs, dropout_factor)

    model_adaptor = build_model_adapter(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = {'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' },
              metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })

    return model


def intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
	boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def tune_test_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset


def save_results(save_path, batch_size, dropout_factor, epochs, avg_correct, avg_IoU):

    data = {
        "save_path": [save_path],
        "batch_size": [batch_size],
        "dropout_factor": [dropout_factor],
        "epochs": [epochs],
        "avg_correct": [avg_correct],
        "avg_IoU": [avg_IoU]
    }

    # Create a DataFrame with the dictionary
    df_current_iteration = pd.DataFrame(data)

    # Check if results.csv already exists
    try:
        df_existing = pd.read_csv("results.csv")

        # Append the new row to the existing DataFrame
        df_existing = pd.concat([df_existing, df_current_iteration], ignore_index=True)

    except FileNotFoundError:
        # If results.csv doesn't exist, create a new DataFrame
        df_existing = df_current_iteration

    # Save the DataFrame back to results.csv
    df_existing.to_csv("results.csv", index=False)


if __name__ == "__main__":

    training_path = "./data/training_files.json"
    validation_path = "./data/validation_files.json"
    test_path = "./data/test_files.json"

    if os.path.exists(training_path) and os.path.exists(validation_path) and os.path.exists(test_path):
        with open(training_path, 'r') as file:
            training_files = json.load(file)
        with open(validation_path, 'r') as file:
            validation_files = json.load(file)
        with open(test_path, 'r') as file:
            test_files = json.load(file)
    else:
        training_files, validation_files, test_files = list_files()

    print(str(len(training_files)) + " training files")
    print(str(len(validation_files)) + " validation files")
    print(str(len(test_files)) + " test files")

    raw_train_ds = data_load(training_files, train_aug=True)
    raw_validation_ds = data_load(validation_files)
    raw_test_ds = data_load(test_files)

    for batch_size in [16, 32]:

        for dropout_factor in [0.3, 0.5, 0.7]:

            for epochs in [20, 100]:

                train_ds = tune_training_ds(raw_train_ds, batch_size)

                validation_ds = tune_validation_ds(raw_validation_ds)

                model = build_model(tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1,)), dropout_factor)

                history = model.fit(train_ds,
                                    steps_per_epoch=(len(training_files) // batch_size),
                                    validation_data=validation_ds, validation_steps=1,
                                    epochs=epochs,
                                    verbose=0)
                
                test_ds = tune_test_ds(raw_test_ds)

                correct_masks = []
                IoUs = []

                test_list = list(test_ds.take(len(test_ds)).as_numpy_iterator())

                for i in range(len(test_list)):

                    image, labels = test_list[i]

                    predictions = model(image)

                    predicted_box = predictions[1][0] * INPUT_SIZE
                    predicted_box = tf.cast(predicted_box, tf.int32)

                    predicted_label = predictions[0][0]

                    actual_label = labels[0][0]
                    actual_box = labels[1][0] * INPUT_SIZE
                    actual_box = tf.cast(actual_box, tf.int32)

                    correct_mask = 0
                    if (predicted_label[0] > 0.5 and actual_label[0] > 0) or (predicted_label[0] < 0.5 and actual_label[0] == 0):
                        correct_mask = 1

                    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())

                    correct_masks.append(correct_mask)
                    IoUs.append(IoU)

                avg_correct = np.mean(correct_masks)
                avg_IoU = np.mean(IoUs)
                print(avg_correct, avg_IoU)

                timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                save_path = f"model/{timestamp}"

                os.makedirs(save_path, exist_ok=True)

                model.save(save_path)

                save_results(save_path, batch_size, dropout_factor, epochs, avg_correct, avg_IoU)

                print(f"Model run with batch_size {batch_size}, dropout_factor {dropout_factor}, and epochs {epochs} completed at {timestamp}.")
