import time
from multiprocessing import Value, Process

# 不使用线程锁（非线程安全）
def worker(val):
    for idx in range(50):
        time.sleep(0.2)
        val.value += 1

# 使用线程锁
def worker_threadSafe(val):
    for idx in range(50):
        time.sleep(0.2)
        with val.get_lock():
            val.value += 1


if __name__ == "__main__":
    v = Value('i', 0)
    p_list = [Process(target=worker, args=(v,)) for i in range(5)]
    for procs in p_list:
        procs.start()
    for procs in p_list:
        procs.join()

    print(v.value)

    v_safe = Value('i', 0)
    p_list = [Process(target=worker_threadSafe, args=(v_safe,)) for i in range(5)]
    for procs in p_list:
        procs.start()
    for procs in p_list:
        procs.join()

    print(v_safe.value)
