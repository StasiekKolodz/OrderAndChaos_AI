import time

start = time.time()
while True:
    now = time.time()
    if (now - start)%10 == 0:
        with open('vm_test.txt', 'w') as f:
            f.write(f'Time: {now - start}')