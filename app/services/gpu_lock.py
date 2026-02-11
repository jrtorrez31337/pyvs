"""Shared GPU lock to prevent concurrent CUDA operations on the same device."""
import threading

# All services running on cuda:0 must acquire this lock before inference.
# This prevents concurrent CUDA operations that cause OOM or corrupt output.
gpu0_lock = threading.Lock()
