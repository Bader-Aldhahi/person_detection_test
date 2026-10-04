import time


for count in range(10001):
    if count:
        time.sleep(2)
    print(count, flush=True)
