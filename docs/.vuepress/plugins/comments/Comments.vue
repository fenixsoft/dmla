<template>
  <div class="comments-section" ref="sectionRef">
    <!-- 标题区域 -->
    <div class="comments-header">
      <h3>💬 讨论</h3>
      <a v-if="issueUrl" :href="issueUrl" target="_blank" rel="noopener">
        在 GitHub 上查看 →
      </a>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="comments-loading">
      加载评论中...
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="comments-error">
      <div class="warning-icon">⚠️</div>
      <p class="error-message">
        {{ error === 'rate_limit' ? 'API 请求次数已达上限，请稍后再试' : '评论暂时无法加载' }}
      </p>
      <p v-if="error === 'rate_limit'">
        <a href="#" @click.prevent="loginWithGitHub">登录 GitHub</a> 获取更高的请求配额
      </p>
      <a :href="issueUrl" target="_blank" class="gh-link">
        在 GitHub 上查看讨论
      </a>
    </div>

    <!-- 评论内容 -->
    <template v-else>
      <!-- 登录提示 -->
      <div v-if="!isLoggedIn" class="login-prompt">
        <button class="login-btn" @click="loginWithGitHub">
          <span>🔐</span>
          <span>使用 GitHub 登录参与讨论</span>
        </button>
      </div>

      <!-- 评论表单 (已登录) -->
      <div v-else class="comment-form">
        <textarea
          v-model="newComment"
          placeholder="写下你的想法... (支持 Markdown)"
          :disabled="submitting"
        ></textarea>
        <div class="actions">
          <button
            class="submit-btn"
            @click="submitComment"
            :disabled="!newComment.trim() || submitting"
          >
            {{ submitting ? '发表中...' : '发表评论' }}
          </button>
        </div>
      </div>

      <!-- 评论列表 -->
      <div class="comments-list">
        <div
          v-for="comment in comments"
          :key="comment.id"
          class="comment-item"
        >
          <div class="comment-header">
            <img
              :src="comment.user.avatar_url"
              :alt="comment.user.login"
              class="comment-avatar"
            />
            <span class="comment-author">{{ comment.user.login }}</span>
            <span class="comment-time">{{ formatTime(comment.created_at) }}</span>
          </div>
          <div class="comment-body" v-html="renderMarkdown(comment.body)"></div>
        </div>

        <div v-if="comments.length === 0" class="no-comments">
          暂无评论，成为第一个评论者吧！
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { usePageFrontmatter } from '@vuepress/client'

// Props
const props = defineProps({
  // GitHub 仓库 (owner/repo)
  repo: {
    type: String,
    default: ''
  },
  // OAuth Client ID
  clientId: {
    type: String,
    default: ''
  },
  // OAuth 代理 URL
  proxyUrl: {
    type: String,
    default: ''
  }
})

// 状态
const loading = ref(true)
const error = ref(null)
const comments = ref([])
const newComment = ref('')
const submitting = ref(false)
const sectionRef = ref(null)
const observer = ref(null)

// 计算属性
const frontmatter = usePageFrontmatter()
const issueNumber = computed(() => frontmatter.value?.issue?.number)
const issueTitle = computed(() => frontmatter.value?.issue?.title)
const articleTitle = computed(() => frontmatter.value?.title)

const repoInfo = computed(() => {
  const repoPath = props.repo || (typeof __COMMENTS_REPO__ !== 'undefined' ? __COMMENTS_REPO__ : '')
  const [owner, repo] = repoPath.split('/')
  return { owner, repo }
})

const issueUrl = computed(() => {
  if (!repoInfo.value.owner || !repoInfo.value.repo) return ''
  if (issueNumber.value) {
    return `https://github.com/${repoInfo.value.owner}/${repoInfo.value.repo}/issues/${issueNumber.value}`
  }
  return `https://github.com/${repoInfo.value.owner}/${repoInfo.value.repo}/issues`
})

const isLoggedIn = computed(() => {
  return !!localStorage.getItem('gh_token')
})

// 缓存管理
const CACHE_PREFIX = 'gh_comments:'
const CACHE_TTL_MEMORY = 30 * 1000    // 30秒
const CACHE_TTL_STORAGE = 5 * 60 * 1000 // 5分钟

const memoryCache = new Map()

function getCache(key) {
  // 检查内存缓存
  const memCached = memoryCache.get(key)
  if (memCached && Date.now() - memCached.timestamp < CACHE_TTL_MEMORY) {
    return memCached
  }

  // 检查 localStorage 缓存
  const storageKey = CACHE_PREFIX + key
  const stored = localStorage.getItem(storageKey)
  if (stored) {
    try {
      const parsed = JSON.parse(stored)
      if (Date.now() - parsed.timestamp < CACHE_TTL_STORAGE) {
        // 提升到内存缓存
        memoryCache.set(key, parsed)
        return parsed
      }
    } catch (e) {
      // 忽略解析错误
    }
  }

  return null
}

function setCache(key, data, etag) {
  const entry = {
    data,
    etag,
    timestamp: Date.now()
  }
  memoryCache.set(key, entry)
  localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(entry))
}

// API 请求
async function fetchComments() {
  if (!issueNumber.value || !repoInfo.value.owner) {
    loading.value = false
    return
  }

  const cacheKey = `issue_${issueNumber.value}`
  const cached = getCache(cacheKey)

  const headers = {
    'Accept': 'application/vnd.github.v3+json'
  }

  // 添加认证
  const token = localStorage.getItem('gh_token')
  if (token) {
    headers['Authorization'] = `token ${token}`
  }

  // 添加 ETag
  if (cached?.etag) {
    headers['If-None-Match'] = cached.etag
  }

  try {
    const response = await fetch(
      `https://api.github.com/repos/${repoInfo.value.owner}/${repoInfo.value.repo}/issues/${issueNumber.value}/comments`,
      { headers }
    )

    // 处理限流
    if (response.status === 403) {
      error.value = 'rate_limit'
      loading.value = false
      return
    }

    // 未修改，使用缓存
    if (response.status === 304) {
      comments.value = cached.data
      loading.value = false
      return
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const etag = response.headers.get('ETag')
    const data = await response.json()

    // 更新缓存
    setCache(cacheKey, data, etag)
    comments.value = data

  } catch (e) {
    // 尝试使用缓存
    if (cached) {
      comments.value = cached.data
    } else {
      error.value = 'fetch_error'
    }
  } finally {
    loading.value = false
  }
}

// OAuth 登录
function loginWithGitHub() {
  const clientId = props.clientId || (typeof __COMMENTS_CLIENT_ID__ !== 'undefined' ? __COMMENTS_CLIENT_ID__ : '')
  if (!clientId) {
    // 直接跳转到 GitHub Issue
    if (issueUrl.value) {
      window.open(issueUrl.value, '_blank')
    }
    return
  }

  const redirectUri = encodeURIComponent(window.location.href)
  const state = Math.random().toString(36).substring(7)

  localStorage.setItem('oauth_state', state)

  const authUrl = `https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&scope=public_repo&state=${state}`
  window.location.href = authUrl
}

// 处理 OAuth 回调
async function handleOAuthCallback() {
  const urlParams = new URLSearchParams(window.location.search)
  const code = urlParams.get('code')
  const state = urlParams.get('state')

  if (!code) return

  // 验证 state
  const savedState = localStorage.getItem('oauth_state')
  if (state !== savedState) {
    console.error('OAuth state mismatch')
    return
  }

  // 清理 URL
  window.history.replaceState({}, document.title, window.location.pathname)

  try {
    // 使用代理服务交换 token
    const proxyUrl = props.proxyUrl || (typeof __COMMENTS_PROXY_URL__ !== 'undefined' ? __COMMENTS_PROXY_URL__ : '')

    if (proxyUrl) {
      const response = await fetch(proxyUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      })
      const data = await response.json()
      if (data.access_token) {
        localStorage.setItem('gh_token', data.access_token)
      }
    } else {
      // 无代理时，提示用户直接在 GitHub 评论
      console.warn('No OAuth proxy configured')
    }
  } catch (e) {
    console.error('OAuth callback error:', e)
  }
}

// 发表评论
async function submitComment() {
  if (!newComment.value.trim() || submitting.value) return

  const token = localStorage.getItem('gh_token')
  if (!token) {
    loginWithGitHub()
    return
  }

  submitting.value = true

  try {
    const response = await fetch(
      `https://api.github.com/repos/${repoInfo.value.owner}/${repoInfo.value.repo}/issues/${issueNumber.value}/comments`,
      {
        method: 'POST',
        headers: {
          'Accept': 'application/vnd.github.v3+json',
          'Authorization': `token ${token}`
        },
        body: JSON.stringify({ body: newComment.value })
      }
    )

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    // 清空输入并刷新评论
    newComment.value = ''

    // 清除缓存
    const cacheKey = `issue_${issueNumber.value}`
    memoryCache.delete(cacheKey)
    localStorage.removeItem(CACHE_PREFIX + cacheKey)

    // 重新加载
    await fetchComments()

  } catch (e) {
    console.error('Submit comment error:', e)
  } finally {
    submitting.value = false
  }
}

// 工具函数
function formatTime(dateStr) {
  const date = new Date(dateStr)
  const now = new Date()
  const diff = now - date

  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  if (hours < 24) return `${hours} 小时前`
  if (days < 30) return `${days} 天前`

  return date.toLocaleDateString('zh-CN')
}

function renderMarkdown(text) {
  // 简单的 Markdown 渲染 (实际项目中应使用完整的 markdown 解析器)
  if (!text) return ''
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}

// 生命周期
onMounted(() => {
  // 处理 OAuth 回调
  handleOAuthCallback()

  // 使用 IntersectionObserver 懒加载
  observer.value = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        fetchComments()
        observer.value.disconnect()
      }
    })
  }, { rootMargin: '100px' })

  if (sectionRef.value) {
    observer.value.observe(sectionRef.value)
  }
})

onUnmounted(() => {
  if (observer.value) {
    observer.value.disconnect()
  }
})
</script>

<style scoped>
.comments-section {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid var(--c-border, #eaecef);
}

.comments-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.comments-header h3 {
  margin: 0;
  font-size: 1.25rem;
}

.comments-loading {
  text-align: center;
  padding: 2rem;
  color: var(--c-text-light, #666);
}

.comments-error {
  padding: 1.5rem;
  text-align: center;
  background: var(--c-bg-light, #f3f4f5);
  border-radius: 8px;
}

.comments-error .warning-icon {
  font-size: 24px;
  margin-bottom: 8px;
}

.comments-error .error-message {
  color: var(--c-text-light, #666);
  margin-bottom: 12px;
}

.comments-error .gh-link {
  display: inline-block;
  margin-top: 8px;
  padding: 8px 16px;
  color: var(--c-brand, #3eaf7c);
  text-decoration: none;
  border: 1px solid var(--c-brand, #3eaf7c);
  border-radius: 6px;
}

.comments-error .gh-link:hover {
  background: var(--c-brand, #3eaf7c);
  color: #fff;
}

.login-prompt {
  text-align: center;
  padding: 2rem;
  background: var(--c-bg-light, #f3f4f5);
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.login-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background: #24292e;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.login-btn:hover {
  background: #2f363d;
}

.comment-form {
  margin-bottom: 1.5rem;
}

.comment-form textarea {
  width: 100%;
  min-height: 100px;
  padding: 12px;
  font-size: 14px;
  border: 1px solid var(--c-border, #eaecef);
  border-radius: 6px;
  resize: vertical;
  background: var(--c-bg, #fff);
  color: var(--c-text, #2c3e50);
  font-family: inherit;
}

.comment-form .actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}

.submit-btn {
  padding: 8px 16px;
  font-size: 14px;
  color: #fff;
  background: var(--c-brand, #3eaf7c);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.submit-btn:hover:not(:disabled) {
  background: var(--c-brand-light, #4abf8a);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.comment-item {
  padding: 1rem;
  border-bottom: 1px solid var(--c-border, #eaecef);
}

.comment-item:last-child {
  border-bottom: none;
}

.comment-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.comment-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
}

.comment-author {
  font-weight: 500;
}

.comment-time {
  font-size: 12px;
  color: var(--c-text-light, #666);
}

.comment-body {
  font-size: 14px;
  line-height: 1.6;
}

.comment-body :deep(code) {
  padding: 2px 6px;
  background: var(--c-bg-light, #f3f4f5);
  border-radius: 3px;
  font-family: monospace;
}

.no-comments {
  text-align: center;
  padding: 2rem;
  color: var(--c-text-light, #666);
}
</style>