#!/usr/bin/env python3
"""
DMLA 进度报告模块

用于在长时间训练任务中向前端报告进度信息。

使用方式:
    from dmla_progress import ProgressReporter

    progress = ProgressReporter(total_steps=100, description="训练 AlexNet")

    for epoch in range(100):
        train_one_epoch()
        progress.update(epoch + 1, message=f"Epoch {epoch+1}: Loss=0.32")

    progress.complete(message="训练完成")
"""

import json
import time
from pathlib import Path
from typing import Optional

# 进度文件路径
PROGRESS_FILE = Path('/workspace/progress.json')


class ProgressReporter:
    """
    进度报告器

    将进度信息写入 progress.json 文件，供宿主机轮询读取并推送到前端。
    """

    def __init__(
        self,
        total_steps: int,
        description: str = "",
        start_step: int = 0
    ):
        """
        初始化进度报告器

        Args:
            total_steps: 总步数（如 epoch 数）
            description: 任务描述
            start_step: 起始步数（默认 0）
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = start_step
        self.start_time = time.time()

        # 写入初始状态
        self._write_progress(
            step=start_step,
            status="starting",
            message=description
        )

    def update(
        self,
        step: int,
        message: str = "",
        extra_data: Optional[dict] = None
    ):
        """
        更新进度

        Args:
            step: 当前步数
            message: 进度消息
            extra_data: 额外数据（如 loss、accuracy 等）
        """
        self.current_step = step
        percent = (step / self.total_steps) * 100 if self.total_steps > 0 else 0

        elapsed = time.time() - self.start_time

        # 估算剩余时间
        if step > 0:
            avg_time_per_step = elapsed / step
            remaining_steps = self.total_steps - step
            estimated_remaining = avg_time_per_step * remaining_steps
        else:
            estimated_remaining = 0

        self._write_progress(
            step=step,
            status="running",
            message=message,
            percent=percent,
            elapsed=elapsed,
            remaining=estimated_remaining,
            extra_data=extra_data
        )

    def reset(
        self,
        total_steps: Optional[int] = None,
        description: Optional[str] = None,
        keep_start_time: bool = False
    ):
        """
        重置进度报告器参数（用于阶段切换）

        Args:
            total_steps: 新的总步数（如不提供则保持原值）
            description: 新的任务描述（如不提供则保持原值）
            keep_start_time: 是否保持起始时间（默认重置）
        """
        if total_steps is not None:
            self.total_steps = total_steps
        if description is not None:
            self.description = description
        if not keep_start_time:
            self.start_time = time.time()
        self.current_step = 0

        # 写入新阶段的初始状态
        self._write_progress(
            step=0,
            status="starting",
            message=self.description
        )

    def complete(self, message: str = "", extra_data: Optional[dict] = None):
        """
        标记任务完成

        Args:
            message: 完成消息
            extra_data: 额外数据（如最终 accuracy 等）
        """
        elapsed = time.time() - self.start_time

        self._write_progress(
            step=self.total_steps,
            status="complete",
            message=message,
            percent=100.0,
            elapsed=elapsed,
            remaining=0,
            extra_data=extra_data
        )

    def error(self, message: str = "", extra_data: Optional[dict] = None):
        """
        标记任务出错

        Args:
            message: 错误消息
            extra_data: 额外数据（如错误详情）
        """
        elapsed = time.time() - self.start_time

        self._write_progress(
            step=self.current_step,
            status="error",
            message=message,
            percent=None,
            elapsed=elapsed,
            remaining=None,
            extra_data=extra_data
        )

    def _write_progress(
        self,
        step: int,
        status: str,
        message: str,
        percent: Optional[float] = None,
        elapsed: float = 0,
        remaining: Optional[float] = None,
        extra_data: Optional[dict] = None
    ):
        """
        写入进度文件

        Args:
            step: 当前步数
            status: 状态（starting, running, complete, error）
            message: 消息
            percent: 百分比
            elapsed: 已用时间（秒）
            remaining: 预计剩余时间（秒）
            extra_data: 额外数据
        """
        data = {
            "description": self.description,
            "total_steps": self.total_steps,
            "current_step": step,
            "percent": percent,
            "message": message,
            "status": status,
            "start_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start_time)),
            "elapsed_seconds": int(elapsed),
            "estimated_remaining": int(remaining) if remaining is not None else None
        }

        # 添加额外数据
        if extra_data:
            data["extra_data"] = extra_data

        # 写入文件
        try:
            PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            # 写入失败不影响训练，仅打印警告
            print(f"Warning: Failed to write progress file: {e}")


def get_progress() -> Optional[dict]:
    """
    读取当前进度

    Returns:
        进度字典，如果文件不存在则返回 None
    """
    try:
        if PROGRESS_FILE.exists():
            content = PROGRESS_FILE.read_text()
            return json.loads(content)
    except Exception:
        pass
    return None


def clear_progress():
    """
    清除进度文件
    """
    try:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
    except Exception:
        pass


# 示例使用
if __name__ == "__main__":
    print("DMLA 进度报告模块示例")
    print("=" * 50)

    progress = ProgressReporter(total_steps=10, description="示例训练任务")

    for i in range(10):
        time.sleep(1)  # 模拟训练
        loss = 0.5 - 0.05 * i
        progress.update(i + 1, message=f"Epoch {i+1}/10", extra_data={"loss": loss})
        print(f"Epoch {i+1}/10 - Loss: {loss:.3f}")

    progress.complete(message="训练完成", extra_data={"final_loss": 0.05})
    print("=" * 50)
    print(f"最终进度: {get_progress()}")
    clear_progress()