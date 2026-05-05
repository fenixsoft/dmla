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
import os
import time
import sys
import threading
import queue
from pathlib import Path
from typing import Optional


def get_progress_file_path() -> Path:
    """
    动态检测进度文件路径

    优先级：
    1. DMLA_PROGRESS_PATH 环境变量（最高优先级，由 Native 模式设置）
    2. Docker 路径 /workspace/progress.json（如果父目录存在）
    3. Native 路径 ~/dmla-data/progress.json（默认）

    Returns:
        进度文件 Path 对象
    """
    # 1. 环境变量优先（Native 模式）
    env_path = os.environ.get('DMLA_PROGRESS_PATH')
    if env_path:
        return Path(env_path)

    # 2. Docker 路径（如果 /workspace 目录存在）
    docker_path = Path('/workspace/progress.json')
    if docker_path.parent.exists():
        return docker_path

    # 3. Native 路径（默认，使用 DMLA_DATA_PATH 或 ~/dmla-data）
    data_path = os.environ.get('DMLA_DATA_PATH')
    if data_path:
        return Path(data_path) / 'progress.json'

    # 最终 fallback：用户主目录下的 dmla-data
    return Path(os.path.expanduser('~')) / 'dmla-data' / 'progress.json'


# 进度文件路径（动态检测）
PROGRESS_FILE = get_progress_file_path()

# stderr 异步写入队列（避免管道阻塞）
_stderr_queue: queue.Queue = queue.Queue()
_stderr_thread: Optional[threading.Thread] = None


def _start_stderr_writer():
    """启动后台 stderr 写入线程（daemon 线程，随主线程退出）"""
    global _stderr_thread
    if _stderr_thread is None or not _stderr_thread.is_alive():
        _stderr_thread = threading.Thread(target=_stderr_writer_loop, daemon=True)
        _stderr_thread.start()


def _stderr_writer_loop():
    """后台线程循环：从队列读取数据并写入 stderr"""
    while True:
        try:
            item = _stderr_queue.get(timeout=1.0)
            if item is None:  # 停止信号
                break
            sys.stderr.write(item)
            sys.stderr.flush()
        except queue.Empty:
            continue  # 队列空，继续等待
        except Exception:
            pass  # 写入失败，忽略（不影响主线程）


def _write_stderr_async(data: str):
    """
    异步写入 stderr（非阻塞）

    将数据放入队列，由后台线程处理写入。
    如果队列积压过多（>100 条），则丢弃旧数据，避免内存爆炸。
    """
    _start_stderr_writer()

    # 如果队列积压过多，清空部分旧数据
    while _stderr_queue.qsize() > 100:
        try:
            _stderr_queue.get_nowait()
        except queue.Empty:
            break

    _stderr_queue.put(data)


class ProgressReporter:
    """
    进度报告器

    将进度信息写入 progress.json 文件，供宿主机轮询读取并推送到前端。
    """

    def __init__(
        self,
        total_steps: int,
        description: str = "",
        start_step: int = 0,
        clear_existing: bool = True
    ):
        """
        初始化进度报告器

        Args:
            total_steps: 总步数（如 epoch 数）
            description: 任务描述
            start_step: 起始步数（默认 0）
            clear_existing: 是否清除已存在的进度文件（默认 True）
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = start_step
        self.start_time = time.time()

        # 清除旧的进度文件，避免显示上一个任务的进度
        if clear_existing:
            clear_progress()

        # 计算初始百分比
        initial_percent = (start_step / total_steps) * 100 if total_steps > 0 else 0

        # 写入初始状态（包含 percent）
        self._write_progress(
            step=start_step,
            status="starting",
            message=description,
            percent=initial_percent
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

        # 写入新阶段的初始状态（包含 percent=0）
        self._write_progress(
            step=0,
            status="starting",
            message=self.description,
            percent=0.0
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
        写入进度信息（stdout + 文件双模式）

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
            "type": "progress",  # 流式消息类型标识
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

        # 1. 文件写入（优先，确保进度数据持久化）
        # 文件写入是可靠的，不受管道阻塞影响
        try:
            PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            # 写入失败不影响训练，仅打印警告
            print(f"Warning: Failed to write progress file: {e}")

        # 2. stderr 异步输出（用于流式 HTTP 响应）
        # 使用后台线程异步写入，避免管道阻塞主线程
        # Windows Docker 环境下管道缓冲区满时会导致阻塞
        try:
            _write_stderr_async(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            # stderr 输出失败不影响训练
            print(f"Warning: Failed to output progress to stderr: {e}")


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