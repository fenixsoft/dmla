const EventEmitter = require('events');

class ChatManager extends EventEmitter {
  constructor() {
    super();
    this.session = null;
    this._pendingResponse = null;
    this._responseBuffer = '';
    this._pendingTimeout = null;
  }

  /**
   * 注册一个对话沙箱会话
   * @param {'docker'|'native'} type - 沙箱类型
   * @param {object} options
   * @param {object} [options.container] - Docker 容器实例
   * @param {object} [options.process] - 子进程实例
   * @param {object} options.stdin - 可写入的 stdin 流
   */
  register(type, { container, process: proc, stdin }) {
    this.session = { type, container, process: proc, stdin, ready: false };
    this._responseBuffer = '';

    // 监听 stdout 解析消息
    const stdout = type === 'native' ? proc.stdout : null;
    // Docker 模式的 stdout 通过 stream 事件处理，不在此处绑定
    if (stdout) {
      stdout.on('data', (data) => this._handleStdout(data));
    }
  }

  /**
   * 标记对话沙箱就绪
   */
  setReady(ready) {
    if (this.session) {
      this.session.ready = ready;
    }
    this.emit('ready', ready);
  }

  /**
   * 查询对话服务状态
   */
  getStatus() {
    if (!this.session) {
      return { ready: false, message: '对话服务未启动' };
    }
    return {
      ready: this.session.ready,
      message: this.session.ready ? '对话服务就绪' : '模型加载中...'
    };
  }

  /**
   * 发送对话消息
   * @param {string} message - 用户消息
   * @returns {Promise<string>} AI 回复
   */
  async send(message) {
    if (!this.session || !this.session.ready) {
      throw new Error('对话服务未就绪');
    }
    if (!this.session.stdin) {
      throw new Error('沙箱 stdin 不可用');
    }

    // 转义消息中的特殊字符，构造安全的 Python 字符串
    const escapedMessage = JSON.stringify(message);
    const code = `print(chat(${escapedMessage}))`;

    return new Promise((resolve, reject) => {
      this._pendingResponse = { resolve, reject, buffer: '' };

      const cmd = JSON.stringify({ action: 'execute', code });
      this.session.stdin.write(cmd + '\n');

      // 超时保护（60秒）
      this._pendingTimeout = setTimeout(() => {
        this._pendingResponse = null;
        reject(new Error('推理超时'));
      }, 60000);
    });
  }

  /**
   * 处理沙箱 stdout 输出（Native 模式）
   */
  _handleStdout(data) {
    const text = data.toString();
    this._responseBuffer += text;

    const lines = this._responseBuffer.split('\n');
    this._responseBuffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        this._handleMessage(msg);
      } catch {
        if (this._pendingResponse) {
          this._pendingResponse.buffer += line + '\n';
        }
      }
    }
  }

  /**
   * 处理单条 JSON 消息
   */
  _handleMessage(msg) {
    switch (msg.type) {
      case 'idle':
        this.setReady(true);
        break;

      case 'pong':
        break;

      case 'stream':
        if (this._pendingResponse) {
          const content = msg.content || msg.text || '';
          if (content) {
            this._pendingResponse.buffer += content;
          }
        }
        break;

      case 'result':
      case 'execute_result':
        if (this._pendingResponse) {
          clearTimeout(this._pendingTimeout);
          const result = msg.content || this._pendingResponse.buffer.trim();
          this._pendingResponse.resolve(result);
          this._pendingResponse = null;
        }
        break;

      case 'error':
        if (this._pendingResponse) {
          clearTimeout(this._pendingTimeout);
          this._pendingResponse.reject(new Error(msg.content || msg.message || '推理出错'));
          this._pendingResponse = null;
        }
        break;
    }
  }

  /**
   * 处理 Docker 模式的流式输出（由 sandbox.js 调用）
   */
  handleDockerStream(data) {
    this._handleStdout(data);
  }

  /**
   * 清除对话沙箱会话
   */
  clear() {
    if (this._pendingResponse) {
      clearTimeout(this._pendingTimeout);
      this._pendingResponse.reject(new Error('沙箱已停止'));
      this._pendingResponse = null;
    }
    this.session = null;
    this._responseBuffer = '';
    this.emit('cleared');
  }
}

// 单例导出
module.exports = new ChatManager();
