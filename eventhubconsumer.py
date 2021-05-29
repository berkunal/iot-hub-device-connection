import logging
import json
import utility
import keras
from keras.utils import np_utils
from sklearn import preprocessing
import pandas as pd
import numpy as np
from azure.eventhub import EventHubConsumerClient

connection_str = 'Endpoint=sb://berlin.servicebus.windows.net/;SharedAccessKeyName=iothubroutes_berlinHub;SharedAccessKey=D3Mp2TGcUz9NEcw8iJruuj3Kkjo1ZbKvvkN8NahNS8I=;EntityPath=berlin_eh'
consumer_group = '$Default'
eventhub_name = 'berlin_eh'

client = EventHubConsumerClient.from_connection_string(
    connection_str, consumer_group, eventhub_name=eventhub_name)

logger = logging.getLogger("azure.eventhub")

logging.basicConfig(filename='berk.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

model = utility.load_trained_model()
model.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])

USER_33_DATA = []
USER_33_CURRENT_INDEX = 0

def prepare_for_prediction(data):
    dataset = pd.DataFrame(columns=["user", "activity", "timestamp", "x", "y", "z"])
    for i in range(80):
        dataset.loc[i] = data[i]

    dataset['ActivityEncoded'] = dataset.apply(utility.label_encoding, axis=1)

    dataset['x'] = utility.feature_normalize(dataset['x'])
    dataset['y'] = utility.feature_normalize(dataset['y'])
    dataset['z'] = utility.feature_normalize(dataset['z'])
    dataset = dataset.round({'x': 6, 'y': 6, 'z': 6})

    xs = dataset['x'].values
    ys = dataset['y'].values
    zs = dataset['z'].values
    reshaped_segment = np.asarray([[xs, ys, zs]], dtype= np.float32).reshape(-1, 80, 3)
    x = reshaped_segment.reshape(reshaped_segment.shape[0], 240)
    x = x.astype("float32")

    global model
    y_pred_test = model.predict(x)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)


    # x_train, y_test = utility.create_segments_and_labels1(dataset,
    #                                                 79,
    #                                                 180,
    #                                                 'ActivityEncoded')
    # y_test = y_test.astype("float32")
    # y_test = np_utils.to_categorical(y_test, 6)
    # max_y_test = np.argmax(y_test, axis=1)

    # print(max_y_test)
    # print(max_y_pred_test)

    # utility.show_confusion_matrix(max_y_test, max_y_pred_test)

    logging.info("Model predicted %d", max_y_pred_test[0])


def handle_messages(msg):
    global USER_33_DATA, USER_33_CURRENT_INDEX
    USER_33_DATA.append(json.loads(msg))
    USER_33_CURRENT_INDEX += 1

    if USER_33_CURRENT_INDEX == 80:
        prepare_for_prediction(USER_33_DATA)

        USER_33_CURRENT_INDEX = 0
        USER_33_DATA = []

def on_event_batch(partition_context, events):
    for event in events:
        # print("Received event from partition: {}.".format(
        #     partition_context.partition_id))
        logging.info("Telemetry received: %s", event.body_as_str())
        # print("Telemetry received: ", event.body_as_str())
        handle_messages(event.body_as_str())
        # print("Properties (set by device): ", event.properties)
        # print("System properties (set by IoT Hub): ", event.system_properties)
    partition_context.update_checkpoint()


with client:
    client.receive_batch(
        on_event_batch=on_event_batch
        # starting_position="-1",  # "-1" is from the beginning of the partition.
    )
