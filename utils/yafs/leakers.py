import csv
import statistics
import warnings
from copy import copy
from yafs.metrics import Metrics


# Default headers in YAFS
YAFS_APP_LOG_HEADERS = [
    "id",
    "type",
    "app",
    "module",
    "message",
    "DES.src",
    "DES.dst",
    "TOPO.src",
    "TOPO.dst",
    "module.src",
    "service",
    "time_in",
    "time_out",
    "time_emit",
    "time_reception",
]
YAFS_LINK_LOG_HEADERS = [
    "id",
    "type",
    "src",
    "dst",
    "app",
    "latency",
    "message",
    "ctime",
    "size",
    "buffer",
]
NO_RESULT_DEFAULT = 9999999.9


class MetricLeaker(Metrics):
    """
    A simple leaker object that saves every feedback received from YAFS. Final result of this object is a tuple of two lists,
    first one being the application logs and the second one being link logs.
    """

    def __init__(self):
        """Initializes the MetricLeaker with empty logs."""
        self.app_log = []  # List to store application logs
        self.link_log = []  # List to store link logs
        self.result = (self.app_log, self.link_log)  # Tuple to store the final result

    def flush(self):
        """Flushes the logs."""
        pass

    def insert(self, value):
        """Inserts a value into the application log.

        Args:
            value (dict): The value to be inserted.
        """
        self.app_log.append(
            copy(value)
        )  # Append a copy of the value to the application log

    def insert_link(self, value):
        """Inserts a value into the link log.

        Args:
            value (dict): The value to be inserted.
        """
        self.app_log.append(copy(value))  # Append a copy of the value to the link log

    def close(self):
        """Closes the leaker and finalizes the result."""
        self.result = (self.app_log, self.link_log)  # Finalize the result


class RunTimeLeaker(Metrics):
    """
    A useful leaker object for when you need to get the total run time for applications. Final result of this leaker is the
    average run time of every app. Run time takes into account the execution time AND transmission time.
    """

    def __init__(self):
        """Initializes the RunTimeLeaker with empty time and counter dictionaries."""
        self.time = {}  # Dictionary to store total time for each app
        self.counter = {}  # Dictionary to store count of occurrences for each app
        self.result = None  # Variable to store the final result

    def flush(self):
        """Flushes the logs."""
        pass

    def insert(self, value):
        """Inserts a value into the time and counter dictionaries.

        Args:
            value (dict): The value to be inserted.
        """
        app_name = value["app"]
        if app_name in self.time:
            self.time[app_name] += (
                value["time_out"] - value["time_emit"]
            )  # Update total time
            if value["module.src"] == "Source_0":
                self.counter[app_name] += 1  # Update count if source is "Source_0"
        else:
            self.time[app_name] = (
                value["time_out"] - value["time_emit"]
            )  # Initialize total time
            self.counter[app_name] = 1  # Initialize count

    def insert_link(self, value):
        """Inserts a value into the link log. This method is not used in RunTimeLeaker.

        Args:
            value (dict): The value to be inserted.
        """
        pass

    def close(self):
        """Closes the leaker and calculates the average run time for each app."""
        times = self.time
        count = self.counter
        stats = [
            times[name] / count[name] for name in times
        ]  # Calculate average run time for each app
        if len(stats) == 0:
            self.result = (
                NO_RESULT_DEFAULT  # Set default result if no stats are available
            )
            warnings.warn(
                "No Results were captured during simulation. Returning default result."
            )
        else:
            self.result = statistics.mean(stats)  # Calculate mean of the stats


class RunTimeLeakerWithOutputs(Metrics):
    """
    A useful leaker object for when you need to get the total run time for applications. Final result of this leaker is the
    average run time of every app. Run time takes into account the execution time AND transmission time.
    This class also writes everything inside a file.
    """

    def __init__(self, result_path: str):
        """Initializes the RunTimeLeakerWithOutputs with empty time and counter dictionaries and opens log files.

        Args:
            result_path (str): The path where the result files will be saved.
        """
        self.time = {}  # Dictionary to store total time for each app
        self.counter = {}  # Dictionary to store count of occurrences for each app
        self.result = None  # Variable to store the final result

        # Open application log file and write headers
        self.__app_log_file = open(f"{result_path}_apps.csv", "w")
        self.__app_log_writer = csv.DictWriter(
            self.__app_log_file, YAFS_APP_LOG_HEADERS
        )
        self.__app_log_writer.writeheader()

        # Open link log file and write headers
        self.__link_log_file = open(f"{result_path}_links.csv", "w")
        self.__link_log_writer = csv.DictWriter(
            self.__link_log_file, YAFS_LINK_LOG_HEADERS
        )
        self.__link_log_writer.writeheader()

    def flush(self):
        """Flushes the log files."""
        self.__app_log_file.flush()  # Flush application log file
        self.__link_log_file.flush()  # Flush link log file

    def insert(self, value):
        """Inserts a value into the time and counter dictionaries and writes to the app log file.

        Args:
            value (dict): The value to be inserted.
        """
        app_name = value["app"]
        if app_name in self.time:
            self.time[app_name] += (
                value["time_out"] - value["time_emit"]
            )  # Update total time
            if value["module.src"] == "Source_0":
                self.counter[app_name] += 1  # Update count if source is "Source_0"
        else:
            self.time[app_name] = (
                value["time_out"] - value["time_emit"]
            )  # Initialize total time
            self.counter[app_name] = 1  # Initialize count

        self.__app_log_writer.writerow(value)

    def insert_link(self, value):
        """Inserts a value into the link log file.

        Args:
            value (dict): The value to be inserted.
        """
        self.__link_log_writer.writerow(value)  # Write value to link log file

    def close(self):
        """Closes the leaker, closes the log files, and calculates the average run time for each app."""
        self.__app_log_file.close()  # Close application log file
        self.__link_log_file.close()  # Close link log file
        times = self.time
        count = self.counter
        stats = [
            times[name] / count[name] for name in times
        ]  # Calculate average run time for each app
        if len(stats) == 0:
            self.result = (
                NO_RESULT_DEFAULT  # Set default result if no stats are available
            )
            warnings.warn(
                "No Results were captured during simulation. Returning default result."
            )
        else:
            self.result = statistics.mean(stats)  # Calculate mean of the stats


class FilteredMetricLeaker(Metrics):
    """
    A leaker object that can leak specified information. You can use this to get specific information from YAFS.
    Final result of this object is a tuple of two lists, first one being the application logs and the second one
    being link logs.
    """

    def __init__(self, app_log_filter: list[str], link_log_filter: list[str]):
        """
        Initializes the FilteredMetricLeaker with specified filters.

        Args:
            app_log_filter (list[str]): A list of fields to be included in the application log.
            link_log_filter (list[str]): A list of fields to be included in the link log.

        app_log_filter is a subset of:
          "id", "type", "app", "module", "message", "DES.src", "DES.dst", "TOPO.src", "TOPO.dst", "module.src", "service",
          "time_in","time_out", "time_emit","time_reception"

        link_log_filter is a subset of:
          "id", "type", "src", "dst", "app", "latency", "message", "ctime", "size", "buffer"
        """
        self.app_log = []  # List to store filtered application logs
        self.link_log = []  # List to store filtered link logs
        self.app_log_filter = ()  # Tuple to store application log filter fields
        self.link_log_filter = ()  # Tuple to store link log filter fields
        if app_log_filter:
            self.app_log_filter = tuple(
                app_log_filter
            )  # Set application log filter fields
        else:
            raise Exception("At least one item is necessary for app_log_filter.")
        if link_log_filter:
            self.link_log_filter = tuple(link_log_filter)  # Set link log filter fields
        else:
            raise Exception("At least one item is necessary for link_log_filter.")

    def flush():
        """Flushes the logs."""
        pass

    def insert(self, value):
        """Inserts a value into the time and counter dictionaries.

        Args:
            value (dict): The value to be inserted.
        """
        self.app_log.append(
            {k: value[k] for k in self.app_log_filter}
        )  # Append filtered value to application log

    def insert_link(self, value):
        """Inserts a value into the link log.

        Args:
            value (dict): The value to be inserted.
        """
        self.link_log.append(
            {k: value[k] for k in self.link_log_filter}
        )  # Append filtered value to link log

    def close(self):
        """Closes the leaker, closes the log files, and calculates the average run time for each app."""
        self.result = (self.app_log, self.link_log)  # Finalize the result
