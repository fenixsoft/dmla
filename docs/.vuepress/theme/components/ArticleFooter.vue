<template>
  <footer class="article-footer">
    <div class="footer-left">
      <div class="footer-meta">
        <div class="meta-item word-count" :title="wordCountHint">
          <svg class="meta-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
          <span class="meta-text">{{ isEnglish ? 'Words: ' : '文章字数：' }}{{ formattedWordCount }}</span>
        </div>
        <div v-if="lastUpdated" class="meta-item update-time">
          <svg class="meta-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="16" y1="2" x2="16" y2="6"></line>
            <line x1="8" y1="2" x2="8" y2="6"></line>
            <line x1="3" y1="10" x2="21" y2="10"></line>
          </svg>
          <span class="meta-text">{{ isEnglish ? 'Updated ' : '更新于 ' }}{{ lastUpdated }}</span>
        </div>
      </div>
    </div>

    <div class="github-star">
      <GithubButton
        href="https://github.com/fenixsoft/dmla"
        data-icon="octicon-star"
        data-show-count="true"
        :data-text="isEnglish ? 'Star' : 'Star'"
        aria-label="Star fenixsoft/dmla on GitHub"
      />
    </div>
  </footer>
</template>

<script setup>
import { computed } from 'vue'
import { usePageData, useRoute } from '@vuepress/client'
import GithubButton from '../../components/GithubButton.vue'

const page = usePageData()
const route = useRoute()
const isEnglish = computed(() => route.path.startsWith('/en/'))

// 字数统计 - VuePress v2 中 wordCount 直接在 page 根级别
const wordCount = computed(() => page.value.wordCount || 0)
const textWordCount = computed(() => page.value.textWordCount || 0)
const codeWordCount = computed(() => page.value.codeWordCount || 0)
const formattedWordCount = computed(() => wordCount.value.toLocaleString())
const wordCountHint = computed(() => {
  const text = textWordCount.value.toLocaleString()
  const code = codeWordCount.value.toLocaleString()
  if (isEnglish.value) {
    return `Text: ${text} chars\nCode: ${code} chars`
  }
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
.footer-left {
  display: flex;
}

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

/* 右侧 GitHub Star 按钮 */
.github-star {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  padding-right: 6px;
}

/* 响应式适配 */
@media (max-width: 719px) {
  .article-footer {
    flex-direction: row;
    align-items: flex-start;
  }

  .github-star { align-items: flex-end; }
}
</style>
