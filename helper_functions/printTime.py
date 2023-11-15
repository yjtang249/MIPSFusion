import datetime


# @brief: 返回当前时间的字符串(格式1)
def printCurrentDatetime():
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S: ')
    return current_datetime


# @brief: 返回当前时间的字符串(格式2)
def getCurrentDatetime():
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return str(current_datetime)