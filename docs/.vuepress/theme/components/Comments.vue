<template>
  <div class="comments-section" ref="sectionRef">
    <!-- 标题区域 -->
    <div class="comments-header">
      <div class="comments-title">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span>讨论</span>
        <span v-if="comments.length" class="comments-count">{{ comments.length }}</span>
      </div>
      <a v-if="issueUrl" :href="issueUrl" target="_blank" rel="noopener" class="gh-link">
        在 GitHub 上查看
        <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
          <polyline points="15 3 21 3 21 9"/>
          <line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
      </a>
    </div>

    <!-- 无 Issue 关联时的提示 -->
    <div v-if="!issueNumber" class="comments-placeholder">
      <svg class="placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <p class="placeholder-text">评论区正在准备中...</p>
      <p class="placeholder-hint">您可以在 <a :href="repoIssuesUrl" target="_blank" rel="noopener">GitHub Issues</a> 中参与讨论</p>
    </div>

    <!-- 加载状态 -->
    <div v-else-if="loading" class="comments-loading">
      加载评论中...
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="comments-error">
      <svg class="warning-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <p class="error-message">
        {{ error === 'rate_limit' ? 'API 请求次数已达上限，请稍后再试' : '评论暂时无法加载' }}
      </p>
      <p v-if="error === 'rate_limit'">
        <a href="#" @click.prevent="loginWithGitHub">登录 GitHub</a> 获取更高的请求配额
      </p>
      <a :href="issueUrl" target="_blank" class="gh-link-btn">
        在 GitHub 上查看讨论
      </a>
    </div>

    <!-- 评论内容 -->
    <template v-else>
      <!-- 评论提示 -->
      <div class="comment-hint">
        <p>评论功能通过 GitHub Issues 实现</p>
        <a :href="issueUrl" target="_blank" rel="noopener" class="comment-link">
          <svg class="github-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
          在 GitHub 上发表评论
        </a>
      </div>

      <!-- 评论列表 -->
      <div class="comments-list">
        <div
          v-for="comment in comments"
          :key="comment.id"
          class="comment-item"
        >
          <div class="comment-header">
            <div
              class="comment-avatar"
              :style="{ background: getAvatarColor(comment.user.login) }"
            >
              {{ comment.user.login.charAt(0).toUpperCase() }}
            </div>
            <div class="comment-meta">
              <span class="comment-author">{{ comment.user.login }}</span>
              <span class="comment-time">{{ formatTime(comment.created_at) }}</span>
            </div>
          </div>
          <div class="comment-body" v-html="renderMarkdown(comment.body)"></div>
        </div>

        <div v-if="comments.length === 0" class="no-comments">
          暂无评论，<a :href="issueUrl" target="_blank" rel="noopener">成为第一个评论者</a>！
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { usePageFrontmatter } from '@vuepress/client'

const props = defineProps({
  repo: {
    type: String,
    default: ''
  },
  clientId: {
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
  const repoPath = props.repo || ''
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

const repoIssuesUrl = computed(() => {
  if (!repoInfo.value.owner || !repoInfo.value.repo) return ''
  return `https://github.com/${repoInfo.value.owner}/${repoInfo.value.repo}/issues`
})

const isLoggedIn = computed(() => {
  return false // 静态站点不支持 OAuth 登录
})

// 缓存管理
const CACHE_PREFIX = 'gh_comments:'
const CACHE_TTL_MEMORY = 30 * 1000
const CACHE_TTL_STORAGE = 5 * 60 * 1000

const memoryCache = new Map()

function getCache(key) {
  const memCached = memoryCache.get(key)
  if (memCached && Date.now() - memCached.timestamp < CACHE_TTL_MEMORY) {
    return memCached
  }

  if (typeof window === 'undefined') return null
  const storageKey = CACHE_PREFIX + key
  const stored = localStorage.getItem(storageKey)
  if (stored) {
    try {
      const parsed = JSON.parse(stored)
      if (Date.now() - parsed.timestamp < CACHE_TTL_STORAGE) {
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
  if (typeof window !== 'undefined') {
    localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(entry))
  }
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

  const token = typeof window !== 'undefined' ? localStorage.getItem('gh_token') : null
  if (token) {
    headers['Authorization'] = `token ${token}`
  }

  if (cached?.etag) {
    headers['If-None-Match'] = cached.etag
  }

  try {
    const response = await fetch(
      `https://api.github.com/repos/${repoInfo.value.owner}/${repoInfo.value.repo}/issues/${issueNumber.value}/comments`,
      { headers }
    )

    if (response.status === 403) {
      error.value = 'rate_limit'
      loading.value = false
      return
    }

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

    setCache(cacheKey, data, etag)
    comments.value = data

  } catch (e) {
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
  // 静态站点无法安全存储 OAuth token，直接跳转到 GitHub Issue 评论
  if (issueUrl.value) {
    window.open(issueUrl.value, '_blank')
  } else if (repoIssuesUrl.value) {
    window.open(repoIssuesUrl.value, '_blank')
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

    newComment.value = ''

    const cacheKey = `issue_${issueNumber.value}`
    memoryCache.delete(cacheKey)
    localStorage.removeItem(CACHE_PREFIX + cacheKey)

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
  if (!text) return ''
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}

function getAvatarColor(name) {
  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
  const index = name.charCodeAt(0) % colors.length
  return colors[index]
}

// 生命周期
onMounted(() => {
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

<style lang="scss" scoped>
@import '../styles/index.scss';

.comments-section {
  margin-top: 32px;
  padding-top: 32px;
  border-top: 1.5px solid #E4E4E7;
}

.comments-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.comments-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #18181B;
}

.comments-count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  background: #EFF6FF;
  color: #2563EB;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 500;
}

.gh-link {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  color: #2563EB;
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
}

.comments-placeholder {
  text-align: center;
  padding: 2rem;
  background: #FAFAFA;
  border-radius: 12px;
}

.placeholder-icon {
  width: 32px;
  height: 32px;
  stroke: #A1A1AA;
  margin-bottom: 12px;
}

.placeholder-text {
  color: #71717A;
  font-size: 14px;
  margin-bottom: 8px;
}

.placeholder-hint {
  color: #A1A1AA;
  font-size: 13px;
}

.placeholder-hint a {
  color: #2563EB;
  text-decoration: none;
}

.placeholder-hint a:hover {
  text-decoration: underline;
}

.comments-loading {
  text-align: center;
  padding: 2rem;
  color: #71717A;
}

.comments-error {
  padding: 1.5rem;
  text-align: center;
  background: #FAFAFA;
  border-radius: 12px;
}

.warning-icon {
  width: 24px;
  height: 24px;
  stroke: #F59E0B;
  margin-bottom: 8px;
}

.error-message {
  color: #71717A;
  margin-bottom: 12px;
}

.gh-link-btn {
  display: inline-block;
  margin-top: 8px;
  padding: 8px 16px;
  color: #2563EB;
  text-decoration: none;
  border: 1px solid #2563EB;
  border-radius: 8px;
  transition: all 0.2s ease;

  &:hover {
    background: #2563EB;
    color: #fff;
  }
}

.comment-hint {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: #F8FAFC;
  border-radius: 12px;
  margin-bottom: 1.5rem;

  p {
    margin: 0;
    color: #64748B;
    font-size: 14px;
  }
}

.comment-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 500;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: #fff;
  background: #24292E;
  border-radius: 8px;
  text-decoration: none;
  transition: background 0.2s ease;

  &:hover {
    background: #32383F;
  }
}

.github-icon {
  width: 16px;
  height: 16px;
}

.comments-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.comment-item {
  padding: 16px 0;
  border-bottom: 1px solid #E4E4E7;

  &:last-child {
    border-bottom: none;
  }
}

.comment-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.comment-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
}

.comment-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.comment-author {
  font-size: 14px;
  font-weight: 500;
  color: #18181B;
}

.comment-time {
  font-size: 12px;
  color: #71717A;
}

.comment-body {
  font-size: 14px;
  line-height: 1.6;
  color: #18181B;

  :deep(code) {
    padding: 2px 6px;
    background: #EFF6FF;
    border-radius: 4px;
    font-family: 'Fira Code', monospace;
    font-size: 13px;
  }
}

.no-comments {
  text-align: center;
  padding: 2rem;
  color: #71717A;
}

.icon {
  width: 20px;
  height: 20px;
}

.icon-sm {
  width: 14px;
  height: 14px;
}

.github-icon {
  width: 16px;
  height: 16px;
}
</style>