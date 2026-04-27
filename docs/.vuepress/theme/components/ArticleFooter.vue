<template>
  <footer class="article-footer">
    <!-- 左侧：字数统计 + 更新时间 -->
    <div class="footer-meta">
      <!-- 字数统计 -->
      <div class="meta-item word-count" :title="wordCountHint">
        <svg class="meta-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        <span class="meta-text">文章字数：{{ formattedWordCount }}</span>
      </div>
      <!-- 更新时间 -->
      <div v-if="lastUpdated" class="meta-item update-time">
        <svg class="meta-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
          <line x1="16" y1="2" x2="16" y2="6"></line>
          <line x1="8" y1="2" x2="8" y2="6"></line>
          <line x1="3" y1="10" x2="21" y2="10"></line>
        </svg>
        <span class="meta-text">更新于 {{ lastUpdated }}</span>
      </div>
    </div>

    <!-- 右侧：GitHub Star 按钮 -->
    <div class="github-star">
      <a
        class="star-button"
        href="https://github.com/fenixsoft/dmla"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg class="star-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
        </svg>
        <span class="star-text">Star</span>
      </a>
      <ClientOnly>
        <span class="star-count">{{ starCount }}</span>
      </ClientOnly>
    </div>
  </footer>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { usePageData } from '@vuepress/client'
import { ClientOnly } from 'vuepress/client'

const page = usePageData()

// 字数统计 - VuePress v2 中 wordCount 直接在 page 根级别
const wordCount = computed(() => page.value.wordCount || 0)
const textWordCount = computed(() => page.value.textWordCount || 0)
const codeWordCount = computed(() => page.value.codeWordCount || 0)
const formattedWordCount = computed(() => wordCount.value.toLocaleString())
const wordCountHint = computed(() => {
  const text = textWordCount.value.toLocaleString()
  const code = codeWordCount.value.toLocaleString()
  return `文字：${text} 字\n代码：${code} 字`
})

// 更新时间
const lastUpdated = computed(() => {
  const timestamp = page.value.git?.updatedTime
  if (!timestamp) return null

  const date = new Date(timestamp)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')

  return `${year}-${month}-${day}`
})

// GitHub Star 数（客户端实时获取）
const starCount = ref('--')

onMounted(async () => {
  try {
    const response = await fetch('https://api.github.com/repos/fenixsoft/dmla')
    if (response.ok) {
      const data = await response.json()
      starCount.value = data.stargazers_count.toLocaleString()
    }
  } catch (e) {
    // API 获取失败时保持 '--'
    console.warn('获取 GitHub Star 数失败:', e)
  }
})
</script>

<style scoped>
.article-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 2rem;
  border-top: 1.5px solid #E4E4E7;
  min-height: 69px;
}

/* 左侧元信息 */
.footer-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-left: 6px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.meta-icon {
  flex-shrink: 0;
}

.meta-text {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.word-count .meta-icon {
  color: #71717A;
}

.word-count .meta-text {
  color: #71717A;
  font-size: 13px;
  font-weight: 400;
}

.update-time .meta-icon {
  color: #A1A1AA;
}

.update-time .meta-text {
  color: #A1A1AA;
  font-size: 12px;
  font-weight: 400;
}

/* 右侧 GitHub Star */
.github-star {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-right: 6px;
}

.star-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: #24292E;
  border-radius: 8px;
  color: #FFFFFF !important;
  text-decoration: none;
  transition: background 0.2s;
}

.star-button:hover {
  background: #32383F;
}

.star-icon {
  flex-shrink: 0;
}

.star-text {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 13px;
  color: #ffffff;
  font-weight: 500;
}

.star-count {
  padding: 6px 12px;
  border: 1px solid #E4E4E7;
  border-radius: 6px;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: #18181B;
}

/* 响应式适配 */
@media (max-width: 719px) {
  .article-footer {
    flex-direction: column;
    gap: 16px;
    align-items: flex-start;
  }

  .github-star {
    width: 100%;
    justify-content: flex-start;
  }
}

@media (max-width: 419px) {
  .star-button {
    padding: 6px 12px;
  }

  .star-text {
    font-size: 12px;
  }
}
</style>