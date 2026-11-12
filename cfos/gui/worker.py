"""后台任务：长时间计算放在子线程里，界面不卡死，可以显示进度和取消。"""
from __future__ import annotations

import traceback

from qtpy.QtCore import QThread, Signal


class Task(QThread):
    progress = Signal(int, int, str)   # 已完成, 总数, 说明
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def report(self, done, total, text=""):
        self.progress.emit(int(done), int(total), text)

    def run(self):
        try:
            res = self.fn(self, *self.args, **self.kwargs)
            self.finished_ok.emit(res)
        except InterruptedError:
            self.failed.emit("任务已取消。")
        except Exception as e:  # noqa: BLE001
            from ..core.log import log, friendly
            log.error("任务出错\n%s", traceback.format_exc())
            self.failed.emit(friendly(e))
