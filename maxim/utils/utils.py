import os
import random
import socket
import threading
import time


def create_cuid_generator():
    counter = 0
    lock = threading.Lock()

    def generate_cuid() -> str:
        """
        Generate a collision-resistant unique identifier (CUID).
        Format: c{timestamp}{counter}{random}{fingerprint}
        """
        nonlocal counter

        # Get timestamp
        timestamp = str(int(time.time() * 1000))[:8]

        # Increment counter (thread-safe)
        with lock:
            counter = (counter + 1) % 1000000
            counter_str = str(counter).zfill(6)

        # Random component
        random_component = str(random.randint(0, 999999)).zfill(6)

        # Fingerprint
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"

        pid = os.getpid()
        hostname_hash = sum(ord(c) for c in hostname) % 100000
        fingerprint = f"{hostname_hash}{pid % 100000}".zfill(10)

        return f"c{timestamp}{counter_str}{random_component}{fingerprint}"

    return generate_cuid
