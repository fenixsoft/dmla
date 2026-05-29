<template>
  <div class="chat-demo">
    <div class="chat-messages" ref="messagesContainer">
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['chat-message', msg.role === 'user' ? 'chat-user' : 'chat-assistant']"
      >
        <div class="chat-role">{{ msg.role === 'user' ? '你' : 'AI' }}</div>
        <div class="chat-content">{{ msg.content }}</div>
      </div>
      <div v-if="loading" class="chat-message chat-assistant">
        <div class="chat-role">AI</div>
        <div class="chat-content chat-typing">思考中...</div>
      </div>
      <div v-if="!ready && messages.length === 0 && !loading" class="chat-hint">
        请先点击上方代码块的「运行」按钮启动对话服务
      </div>
    </div>
    <div class="chat-input-area">
      <input
        v-model="inputText"
        class="chat-input"
        :disabled="!ready || loading"
        :placeholder="ready ? '输入消息，按 Enter 发送' : '对话服务未启动'"
        @keyup.enter="sendMessage"
      />
      <button
        class="chat-send-btn"
        :disabled="!ready || loading || !inputText.trim()"
        @click="sendMessage"
      >
        发送
      </button>
    </div>
  </div>
</template>

<script>
import { getSandboxEndpoint } from '../plugins/runnable-code/sandbox-config.js'

export default {
  name: 'ChatDemo',
  data() {
    return {
      ready: false,
      loading: false,
      inputText: '',
      messages: [],
      pollTimer: null
    }
  },
  mounted() {
    this.checkStatus()
    this.pollTimer = setInterval(this.checkStatus, 3000)
  },
  beforeDestroy() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer)
      this.pollTimer = null
    }
    this._destroyed = true
  },
  methods: {
    async checkStatus() {
      if (this.ready || this._destroyed) return
      try {
        const endpoint = getSandboxEndpoint()
        const res = await fetch(endpoint + '/api/chat/status')
        if (this._destroyed) return
        const data = await res.json()
        this.ready = data.ready
      } catch {
        if (!this._destroyed) this.ready = false
      }
    },
    async sendMessage() {
      const text = this.inputText.trim()
      if (!text || !this.ready || this.loading || this._destroyed) return

      this.messages.push({ role: 'user', content: text })
      this.inputText = ''
      this.loading = true
      this.scrollToBottom()

      try {
        const endpoint = getSandboxEndpoint()
        const res = await fetch(endpoint + '/api/chat/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        })
        const data = await res.json()
        if (data.response) {
          this.messages.push({ role: 'assistant', content: data.response })
        } else if (data.error) {
          this.messages.push({ role: 'assistant', content: `[错误] ${data.error}` })
        }
      } catch (err) {
        this.messages.push({ role: 'assistant', content: `[请求失败] ${err.message}` })
      } finally {
        this.loading = false
        this.scrollToBottom()
      }
    },
    scrollToBottom() {
      this.$nextTick(() => {
        const container = this.$refs.messagesContainer
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      })
    }
  }
}
</script>

<style scoped>
.chat-demo {
  border: 1px solid var(--c-border, #e2e8f0);
  border-radius: 8px;
  overflow: hidden;
  margin: 16px 0;
}

.chat-messages {
  height: 320px;
  overflow-y: auto;
  padding: 12px;
  background: var(--c-bg-light, #f8fafc);
}

.chat-message {
  margin-bottom: 12px;
  display: flex;
  flex-direction: column;
}

.chat-user {
  align-items: flex-end;
}

.chat-assistant {
  align-items: flex-start;
}

.chat-role {
  font-size: 12px;
  color: var(--c-text-lighter, #94a3b8);
  margin-bottom: 4px;
}

.chat-content {
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.chat-user .chat-content {
  background: var(--c-brand, #3b82f6);
  color: #fff;
  border-bottom-right-radius: 2px;
}

.chat-assistant .chat-content {
  background: var(--c-bg, #fff);
  border: 1px solid var(--c-border, #e2e8f0);
  border-bottom-left-radius: 2px;
}

.chat-typing {
  color: var(--c-text-lighter, #94a3b8);
  font-style: italic;
}

.chat-hint {
  text-align: center;
  color: var(--c-text-lighter, #94a3b8);
  font-size: 13px;
  padding: 16px;
}

.chat-input-area {
  display: flex;
  border-top: 1px solid var(--c-border, #e2e8f0);
  padding: 8px;
  background: var(--c-bg, #fff);
}

.chat-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--c-border, #e2e8f0);
  border-radius: 4px;
  font-size: 14px;
  outline: none;
  background: var(--c-bg-light, #f8fafc);
}

.chat-input:focus {
  border-color: var(--c-brand, #3b82f6);
}

.chat-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chat-send-btn {
  margin-left: 8px;
  padding: 8px 16px;
  background: var(--c-brand, #3b82f6);
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.chat-send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chat-send-btn:hover:not(:disabled) {
  opacity: 0.9;
}
</style>
