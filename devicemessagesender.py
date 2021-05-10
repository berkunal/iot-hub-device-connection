import os
import time
import asyncio
import fileparser
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message

MSG_TXT = '{{"x": {x},"y": {y},"z": {z}}}'
MSG_INTERVAL = 1/24 # Sample rate of the sensor


async def main():

    # Get the data list from our library
    dataList = fileparser.get_data_list_from_file(
        'dataset-cse591\\data\\subject1\\shoulderElevation\\trainingAndPrediction\\prediction_otherClassesBeforeAndAfter15ConsecutiveShoulderElevations\\1.log')

    # Fetch the connection string from an enviornment variable
    conn_str = "HostName=cse591.azure-devices.net;DeviceId=shoulder-sensor-subject-1;SharedAccessKey=Z9ESJBeVapudNU1tHECQCnMg/EtxSjVJf3oSu5YrPxQ="

    # Create instance of the device client using the connection string
    device_client = IoTHubDeviceClient.create_from_connection_string(conn_str)

    # Connect the device client.
    await device_client.connect()

    # Loop through the data and send to IoT Hub
    for data in dataList:
        msg_txt_formatted = MSG_TXT.format(x=data.x, y=data.y, z=data.z)
        message = Message(msg_txt_formatted)

        print( "Sending message: {}".format(message) )
        await device_client.send_message(message)
        print ( "Message successfully sent" )
        time.sleep(MSG_INTERVAL)

    await device_client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
