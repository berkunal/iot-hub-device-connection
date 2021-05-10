import logging
from azure.eventhub import EventHubConsumerClient

connection_str = 'Endpoint=sb://berlineventhub.servicebus.windows.net/;SharedAccessKeyName=iothubroutes_cse591;SharedAccessKey=5+as4J3y7uYUAoMWAi3465aQqIjHt6VE8BZnrm9i1+0=;EntityPath=berlineventhub'
consumer_group = '$Default'
eventhub_name = 'berlineventhub'

client = EventHubConsumerClient.from_connection_string(
    connection_str, consumer_group, eventhub_name=eventhub_name)

logger = logging.getLogger("azure.eventhub")
logging.basicConfig(level=logging.INFO)


def on_event_batch(partition_context, events):
    for event in events:
        print("Received event from partition: {}.".format(
            partition_context.partition_id))
        print("Telemetry received: ", event.body_as_str())
        print("Properties (set by device): ", event.properties)
        print("System properties (set by IoT Hub): ", event.system_properties)
        print()
    partition_context.update_checkpoint()


with client:
    client.receive_batch(
        on_event_batch=on_event_batch
        # starting_position="-1",  # "-1" is from the beginning of the partition.
    )
