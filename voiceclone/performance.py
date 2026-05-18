import time
import psutil

def start_timer():
    return time.time()

def stop_timer(start):
    return round(time.time() - start, 2)

def cpu_usage():
    return psutil.cpu_percent(interval=0.1)