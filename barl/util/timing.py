"""
Timing utilities.
"""

import time
import datetime
import logging


class Timer(object):
    """
    Timer class. Thanks to Eli Bendersky, Josiah Yoder, Jonas Adler, Can Kavaklıoğlu,
    and others from https://stackoverflow.com/a/50957722.
    """

    def __init__(self, name=None, filename=None, level=None, verbose=True):
        self.name = name
        self.filename = filename
        self.level = logging.INFO if level is None else level
        # CHANGES @STELLA: added time_elapsed variable
        self.time_elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.tstart = time.time()
        # CHANGES @STELLA: added return value
        return self

    def __exit__(self, type, value, traceback):
        # CHANGES @STELLA: save time elapsed
        self.time_elapsed = time.time() - self.tstart
        message = "Elapsed: %.2f seconds" % (self.time_elapsed)
        if self.name:
            message = "*[TIME] [%s] " % self.name + message
        if self.verbose: logging.log(self.level, message)
        if self.filename:
            with open(self.filename, "a") as file:
                print(str(datetime.datetime.now()) + ": ", message, file=file)
        
