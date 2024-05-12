import os


class SensorDataParser():
    """Sensor data parser that handles data output from the remote client
    """
    def __init__(self, file_name=None):
        """Configure required dataholders

        Args:
            file_name (str, optional): Name of sensor data csv. Defaults to None.
        """
        self.__file_name = file_name
        self.__data = {}
        self.parse()

    def parse(self):
        """Parse the csv file and split the data into columns
        """
        if os.path.exists(self.__file_name):
            lines = open(self.__file_name, "r").read().strip().split("\n")
            columns = lines[0].split(",")
            for col in columns:
                self.__data[col] = []
            lines = lines[1:]
            for line in lines:
                values = line.split(",")
                for i, val in enumerate(values):
                    try:
                        self.__data[columns[i]].append(round(float(val), 1))
                    except Exception as e:
                        print(f"SDP ERROR: {e}")
        else:
            self.__file_name = None

    def getStartTime(self):
        if self.__data!={}:
            return self.__data[list(self.__data.keys())[0]][0]
        else:
            return 0

    def normalizeTime(self):
        """Normalize the time so that the first value is 0 and the others are
        relative to that value
        """
        if self.__data!={}:
            first_time=self.__data[list(self.__data.keys())[0]][0]
            for i in range(0,len(self.__data[list(self.__data.keys())[0]])):
                self.__data[list(self.__data.keys())[0]][i]-=first_time

    def getSensorNames(self):
        """Get the sensor names
        """
        return list(self.__data.keys())

    def getData(self):
        return self.__data

    def getAverage(self, sensor_name):
        return sum(self.__data[sensor_name])/len(self.__data[sensor_name])

    def getSensorData(self, sensor_name, time):
        """Get the value of a sensor at a given time.

        Args:
            sensor_name (str): Name of the sensor.
            time (float): Time desired.

        Returns:
            float: Value of the sensor at that time.
        """
        if self.__file_name is None:
            return "NaN"
        if sensor_name not in self.__data:
            return f"Sensor '{sensor_name}' not found"
        sensor_values = self.__data[sensor_name]
        time_values=self.__data['Time']
        for i in range(len(time_values)):
            if time==time_values[i]:
                #time=reading time
                return sensor_values[i]
            if time < time_values[i]:
                # time is between reading times,
                # linearly interpolate value
                x0, x1 = time_values[i-1], time_values[i]
                y0, y1 = sensor_values[i-1], sensor_values[i]
                interpolated_value = y0 + (y1 - y0) * ((time - x0) / (x1 - x0))
                return round(interpolated_value, 3)
            if i == len(sensor_values) - 1:
                # time is beyond read times
                return sensor_values[-1]

if __name__ == "__main__":
    # generate random data file
    out = """time,temp,humid\n1,11,17\n2,12,18\n3,13,19\n6,16,22"""
    open("sensor_data.csv","w").write(out)
    # Example usage:
    parser = SensorDataParser("sensor_data.csv")
    print(parser.getSensorData("temp", 4.123))
    print(parser.getData())
    # print(parser.getSensorData("humid", 3))
