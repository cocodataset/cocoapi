import logging
from datetime import datetime
from pytz import utc, timezone


def config_logger(path):
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Shanghai")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.basicConfig()
    logging.getLogger().handlers.pop()

    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    formatter.converter = custom_time

    logging.getLogger().setLevel(logging.INFO)

    log_file_save_name = path
    file_handler = logging.FileHandler(filename=log_file_save_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)