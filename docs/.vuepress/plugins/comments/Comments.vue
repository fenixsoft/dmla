<template>
  <div class="comments-section" ref="commentsContainer">
    <!-- OAuth 错误显示 -->
    <div v-if="oauthError" class="oauth-error">
      <div class="oauth-error-title">GitHub 登录失败</div>
      <div class="oauth-error-message">{{ oauthError.description }}</div>
      <div class="oauth-error-code">错误代码: {{ oauthError.error }}</div>
      <a v-if="oauthError.uri" :href="oauthError.uri" target="_blank" class="oauth-error-link">查看帮助文档</a>
      <button @click="clearError" class="oauth-error-dismiss">关闭</button>
    </div>
    <div id="gitalk-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import Gitalk from 'gitalk'
import 'gitalk/dist/gitalk.css'

const route = useRoute()
const commentsContainer = ref(null)
const oauthError = ref(null)

// 保存原始 console.log
let originalConsoleLog = null

// 隐藏 Gitalk 错误元素
const hideGitalkError = () => {
  const gtError = document.querySelector('.gt-error')
  if (gtError) {
    gtError.style.display = 'none'
  }
}

// 拦截 console.log 捕获 Gitalk OAuth 错误
const setupErrorCapture = () => {
  originalConsoleLog = console.log
  console.log = (...args) => {
    // 检查是否是 Gitalk OAuth 错误日志
    if (args[0] === 'res.data err:' && args[1] && args[1].error) {
      oauthError.value = {
        error: args[1].error,
        description: args[1].error_description || '未知错误',
        uri: args[1].error_uri
      }
      // 延迟隐藏 Gitalk 错误（等待 DOM 更新）
      setTimeout(hideGitalkError, 100)
    }
    originalConsoleLog.apply(console, args)
  }
}

// 恢复原始 console.log
const restoreConsole = () => {
  if (originalConsoleLog) {
    console.log = originalConsoleLog
    originalConsoleLog = null
  }
}

// 初始化 Gitalk（只在客户端执行）
const initGitalk = () => {
  // 确保 title 不为空，使用 pathname 作为后备
  const pageTitle = document.title || location.pathname
  const pageUrl = location.href

  const gitalk = new Gitalk({
    clientID: 'Ov23liQbtfoZGMcsU9VV',
    repo: 'ideaspaces',
    owner: 'fenixsoft',
    admin: ['fenixsoft'],
    id: location.pathname,
    title: pageTitle,
    body: pageUrl,
    labels: ['Gitalk', 'Comment'],
    distractionFreeMode: false,
    proxy: 'https://cros.icyfenix.cn/',
    createIssueManually: false,
    pagerDirection: 'last',
    enableHotKey: true
  })
  gitalk.render('gitalk-container')
}

const clearError = () => {
  oauthError.value = null
}

onMounted(() => {
  setupErrorCapture()
  initGitalk()
})

onBeforeUnmount(() => {
  restoreConsole()
})

// 路由变化时重新渲染
watch(() => route.path, () => {
  const container = document.getElementById('gitalk-container')
  if (container) {
    container.innerHTML = ''
  }
  oauthError.value = null
  initGitalk()
})
</script>

<style scoped>
.comments-section {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid var(--c-border, #eaecef);
}

/* 当显示 OAuth 错误时，隐藏 Gitalk 的错误信息 */
.comments-section:has(.oauth-error) :deep(.gt-error) {
  display: none;
}

.oauth-error {
  background: #FEF2F2;
  border: 1px solid #FECACA;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.oauth-error-title {
  font-weight: 600;
  color: #DC2626;
  margin-bottom: 8px;
}

.oauth-error-message {
  color: #7F1D1D;
  margin-bottom: 8px;
}

.oauth-error-code {
  font-size: 12px;
  color: #991B1B;
  margin-bottom: 8px;
}

.oauth-error-link {
  color: #2563EB;
  font-size: 14px;
  margin-right: 16px;
}

.oauth-error-dismiss {
  background: transparent;
  border: 1px solid #D1D5DB;
  border-radius: 4px;
  padding: 4px 12px;
  font-size: 14px;
  cursor: pointer;
  margin-top: 8px;
}

.oauth-error-dismiss:hover {
  background: #F3F4F6;
}
</style>