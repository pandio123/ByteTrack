import threading

lock = threading.Lock()
condition = threading.Condition(lock)