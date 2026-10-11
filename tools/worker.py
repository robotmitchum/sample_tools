# coding:utf-8
"""
    :module: worker.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.01
"""
from PySide6.QtCore import QObject, Signal, QRunnable


class WorkerSignals(QObject):
    progress = Signal(int)
    progress_range = Signal(int, int)
    message = Signal(str)
    finished = Signal()
    result = Signal(object)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up

    :param callback: The function callback to run on this worker thread
    Supplied args and kwargs will be passed through to the runner
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.stop_requested = False
        self.running = False

        # Add progress emitter to kwargs if absent
        if 'progress_callback' not in self.kwargs:
            self.kwargs['progress_callback'] = self.signals.progress
        if 'message_callback' not in self.kwargs:
            self.kwargs['message_callback'] = self.signals.message

    def run(self):
        """
        Execute the function and emit progress updates
        """
        if not self.running:
            self.running = True
            try:
                result = self.fn(self, *self.args, **self.kwargs)  # Execute the callback function
                self.signals.result.emit(result)
            except Exception as e:
                print(f'Error: {e}')
            finally:
                self.running = False
                self.signals.finished.emit()

    def request_stop(self):
        self.stop_requested = True

    def is_stopped(self, progress_callback=None, message_callback=None):
        if self.stop_requested:
            if progress_callback is not None:
                progress_callback.emit(0)
                message_callback.emit('Process stopped by user')
            return True
        return False
