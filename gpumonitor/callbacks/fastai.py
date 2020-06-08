from fastai2.callback.all import *

from gpumonitor.monitor import GPUStatMonitor


class FastaiGpuMonitorCallback(Callback):
    def __init__(self, delay=1, display_options=None):
        super(FastaiGpuMonitorCallback, self).__init__()
        self.delay = delay
        self.display_options = display_options if display_options else {}

    def begin_epoch(self):
        self.monitor = GPUStatMonitor(self.delay, self.display_options)

    def after_epoch(self):
        self.monitor.stop()
        print("")
        self.monitor.display_average_stats_per_gpu()
