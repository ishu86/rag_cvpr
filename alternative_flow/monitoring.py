import psutil
import time
from functools import wraps

def monitor_resources(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        start_time = time.time()
        start_mem = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent(interval=None)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End monitoring
        end_time = time.time()
        end_mem = psutil.virtual_memory().used
        end_cpu = psutil.cpu_percent(interval=None)
        
        # Log metrics
        print(f"\n=== Resource Monitoring for {func.__name__} ===")
        print(f"Execution time: {end_time - start_time:.2f}s")
        print(f"Memory used: {(end_mem - start_mem)/1024/1024:.2f} MB")
        print(f"CPU usage: {end_cpu - start_cpu:.2f}%")
        print("=" * 40)
        
        return result
    return wrapper