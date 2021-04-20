import re


class Data:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def getDataListFromFile(fileName):
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
                    # print('Shoulder elevation x: ' + data.x +
                    #       ' y: ' + data.y + ' z: ' + data.z)

        return dataList
