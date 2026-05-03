```python runnable gpu timeout=unlimited
import time
from dmla_progress import ProgressReporter

progress = ProgressReporter(total_steps=10, description="进度条50%停顿测试")

for i in range(5):
    time.sleep(1)
    progress.update(i + 1, message=f"已完成步骤 {i+1}/10")
    print(f"步骤 {i+1} 完成")

print("进度条停顿在50%位置...")
time.sleep(3)  # 停顿3秒让用户看到进度条

# 抛出异常测试错误处理
# raise RuntimeError("测试异常：在50%位置抛出错误")

# 以下代码不会执行
for i in range(5, 10):
    time.sleep(1)
    progress.update(i + 1, message=f"已完成步骤 {i+1}/10")
    print(f"步骤 {i+1} 完成")

progress.complete(message="进度条测试完成")
print("测试完成")
```