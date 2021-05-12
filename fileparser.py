import re


class Data:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class DataWISDM:
    def __init__(self, user, activity, timestamp, x, y, z):
        self.user = user
        self.activity = activity
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z


def get_data_list_from_wisdm(fileName):
    with open(fileName, 'r') as reader:

        dataList = []
        i = 0

        for line in reader.readlines():
            user, activity, timestamp, x, y, z = line.split(';')[0].split(',')
            data = DataWISDM(user, activity, timestamp, x, y, z)
            dataList.append(data)

            # End Of File
            if line == '\n':
                print('Done parsing the file')
                break

            i += 1

            if i == 10:
                return dataList


def get_data_list_from_file(fileName):
    with open(fileName, 'r') as reader:

        dataList = []
        logsHaveStarted = False
        firstLog = True

        for line in reader.readlines():

            # Skip first few lines
            if re.search('%~>.*,.*,.*', line):
                logsHaveStarted = True

            if logsHaveStarted:

                # Detect markers
                if re.search(r'\*NOTE.*', line):
                    print('This is a marker!')

                # End Of File
                elif line == '\n':
                    print('Done parsing the file')
                    break

                else:
                    if firstLog:
                        # Get rid of the first 4 chars of first log
                        line = line[3:]
                        firstLog = False

                    x, y, z = line.split(', ')

                    data = Data(x, y, z)
                    dataList.append(data)

        return dataList
