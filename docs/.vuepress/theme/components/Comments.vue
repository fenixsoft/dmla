<template>
  <div class="comments-section">
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
import { usePageFrontmatter } from '@vuepress/client'
import Gitalk from 'gitalk'
import 'gitalk/dist/gitalk.css'

const route = useRoute()
const frontmatter = usePageFrontmatter()
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

const props = defineProps({
  repo: {
    type: String,
    default: 'ideaspaces'
  },
  owner: {
    type: String,
    default: 'fenixsoft'
  },
  admin: {
    type: Array,
    default: () => ['fenixsoft']
  },
  clientId: {
    type: String,
    default: 'Ov23liQbtfoZGMcsU9VV'
  },
  proxy: {
    type: String,
    default: 'https://cros.icyfenix.cn/'
  }
})

// 初始化 Gitalk（只在客户端执行）
const initGitalk = () => {
  // 使用 frontmatter 标题（不带站点名后缀），与 sync-issues.js 保持一致
  const pageTitle = frontmatter.value?.title || frontmatter.value?.issue?.title || location.pathname
  const pageUrl = location.href

  // 从 frontmatter 获取 issue number（如果存在）
  const issueNumber = frontmatter.value?.issue?.number
  const useNumber = typeof issueNumber === 'number' && issueNumber > 0 ? issueNumber : -1

  const gitalk = new Gitalk({
    clientID: props.clientId,
    repo: props.repo,
    owner: props.owner,
    admin: props.admin,
    id: location.pathname,
    number: useNumber,
    title: pageTitle,
    body: pageUrl,
    labels: ['Gitalk', 'Comment'],
    distractionFreeMode: false,
    proxy: props.proxy,
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

<style lang="scss" scoped>
.comments-section {
  max-width: var(--content-width);
  margin: 0 auto;
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

:deep(.gt-container) {
  .gt-header {
    border-bottom: 1px solid #E4E4E7;
    padding-bottom: 12px;
  }

  .gt-btn {
    background-color: #24292E;
    border-color: #24292E;

    &:hover {
      background-color: #32383F;
      border-color: #32383F;
    }
  }

  .gt-btn-preview {
    background-color: #fff;
    color: #24292E;
    border-color: #E4E4E7;

    &:hover {
      background-color: #F6F8FA;
    }
  }

  .gt-header-textarea {
    border-color: #E4E4E7;
    border-radius: 8px;

    &:focus {
      border-color: #2563EB;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
  }

  .gt-comment-content {
    border-radius: 8px;
    border: 1px solid #E4E4E7;
  }

  .gt-comment-body {
    color: #18181B;
  }

  .gt-meta,
  .gt-header-controls-tip {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: #71717A;
  }
}

/* 响应式适配 - 与 VuePress 默认主题一致 */
@media (max-width: 959px) {
  .comments-section {
    padding: 32px 1rem 0;
  }
}

@media (max-width: 719px) {
  .comments-section {
    padding: 24px 1rem 0;
    margin-top: 24px;
  }
}

@media (max-width: 419px) {
  .comments-section {
    padding: 16px 0.5rem 0;
    margin-top: 16px;
  }
}
</style>