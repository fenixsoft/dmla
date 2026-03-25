---
title: 装饰器详解
date: 2024-03-24
tags: [python, advanced]
---

# Python 装饰器详解

装饰器是 Python 中一个非常强大且优雅的特性，它允许我们在不修改原函数代码的情况下，扩展函数的功能。

## 什么是装饰器

装饰器本质上是一个函数，它接受一个函数作为参数，并返回一个新的函数。

```python runnable
# 最简单的装饰器示例
def my_decorator(func):
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# 测试
say_hello()
```

## 带参数的装饰器

```python runnable
def log_execution(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        print(f"参数: args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"返回值: {result}")
        return result
    return wrapper

@log_execution
def add(a, b):
    return a + b

# 测试
add(3, 5)
```

## 装饰器执行流程

```mermaid
graph TD
    A[定义装饰器函数] --> B[定义被装饰函数]
    B --> C[@decorator 语法糖]
    C --> D[装饰器包装原函数]
    D --> E[返回包装后的函数]
    E --> F[调用时执行包装函数]
    F --> G[执行原函数逻辑]
```

## 常用装饰器示例

### 计时装饰器

```python runnable
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"执行时间: {end - start:.4f} 秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()
```

### 缓存装饰器

```python runnable
def memoize(func):
    cache = {}

    def wrapper(*args):
        if args in cache:
            print(f"从缓存获取: {args}")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试缓存效果
print(f"fib(10) = {fibonacci(10)}")
print(f"fib(10) = {fibonacci(10)}")  # 第二次会从缓存获取
```

## 类装饰器

```python runnable
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"第 {self.count} 次调用")
        return self.func(*args, **kwargs)

@CountCalls
def say_hi():
    print("Hi!")

say_hi()
say_hi()
say_hi()
```

## 总结

装饰器的核心概念：

1. **闭包**: 装饰器利用闭包保存原函数引用
2. **高阶函数**: 装饰器接受函数作为参数，返回新函数
3. **@语法糖**: `@decorator` 等价于 `func = decorator(func)`
4. **保留元数据**: 使用 `@functools.wraps` 保留原函数信息