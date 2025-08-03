import threading
from typing import TypeVar, Generic, Optional
from contextlib import contextmanager

T = TypeVar('T')


class ReadWriteLock:
    """
    Reader-writer lock implementation for thread-safe access.
    It prevents data races as specified in the task.
    """

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.RLock())
        self._write_ready = threading.Condition(threading.RLock())

    @contextmanager
    def read_lock(self):
        """Acquire read lock for safe reading operations"""
        self._read_ready.acquire()
        try:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

        try:
            yield
        finally:
            self._read_ready.acquire()
            try:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()
            finally:
                self._read_ready.release()

    @contextmanager
    def write_lock(self):
        """Acquire write lock for safe writing operations"""
        self._write_ready.acquire()
        try:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers += 1
        finally:
            self._write_ready.release()

        try:
            yield
        finally:
            self._write_ready.acquire()
            try:
                self._writers -= 1
                self._write_ready.notify_all()
                self._read_ready.acquire()
                try:
                    self._read_ready.notify_all()
                finally:
                    self._read_ready.release()
            finally:
                self._write_ready.release()


class ThreadSafeDict(Generic[T]):
    """
    Thread-safe dictionary implementation with read-write locks.

    This is REQUIRED to store libraries, documents, and chunks safely
    without data races during concurrent access.
    """

    def __init__(self):
        self._data: dict = {}
        self._lock = ReadWriteLock()

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get item with read lock"""
        with self._lock.read_lock():
            return self._data.get(key, default)

    def set(self, key: str, value: T) -> None:
        """Set item with write lock"""
        with self._lock.write_lock():
            self._data[key] = value

    def delete(self, key: str) -> bool:
        """Delete item with write lock"""
        with self._lock.write_lock():
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists with read lock"""
        with self._lock.read_lock():
            return key in self._data

    def keys(self) -> list:
        """Get all keys with read lock"""
        with self._lock.read_lock():
            return list(self._data.keys())

    def values(self) -> list:
        """Get all values with read lock"""
        with self._lock.read_lock():
            return list(self._data.values())

    def items(self) -> list:
        """Get all items with read lock"""
        with self._lock.read_lock():
            return list(self._data.items())

    def clear(self) -> None:
        """Clear all items with write lock"""
        with self._lock.write_lock():
            self._data.clear()

    def size(self) -> int:
        """Get size with read lock"""
        with self._lock.read_lock():
            return len(self._data)