<template>
  <div class="global-toc">
    <!-- 统计信息 -->
    <div v-if="level === 0" class="stats-info">
      <p>共计 <strong>{{ tocArticleCount }}</strong> 篇文章，合计 <strong>{{ totalWords.toLocaleString() }}</strong> 字，最后更新日期 <strong>{{ lastUpdateDate }}</strong>。</p>
    </div>
    <ol>
      <li v-for="(page, index) in information" :key="index">
        <span v-if="page.links != null">
          <a :href="page.links">
            <span :class="'level' + level">{{ page.title }}</span>
          </a>
          <span class="words">{{ page.words }}</span>
        </span>
        <span v-else :class="'level' + level">
          {{ page.title }}
          <span class="words">{{ page.words }}</span>
        </span>
        <GlobalTOC v-if="page.children && page.children.length > 0"
          :pages="page.children"
          :level="level + 1" />
      </li>
    </ol>
  </div>
</template>

<script>
import { computed, defineComponent, inject } from 'vue'
import { useThemeLocaleData } from '@vuepress/theme-default/lib/client/composables/useThemeData.js'

export default defineComponent({
  name: 'GlobalTOC',
  props: {
    pages: {
      type: [Array, String],
      default: '/'
    },
    level: {
      type: Number,
      default: 0
    }
  },
  setup(props) {
    // 从 Vue app 的 provide 获取全局字数数据
    const wordCountData = inject('wordCountData', {})

    // 从主题配置中动态获取 sidebar 配置
    const themeLocale = useThemeLocaleData()
    const sidebarConfig = computed(() => themeLocale.value.sidebar || [])

    // 获取页面信息
    const information = computed(() => {
      if (!props.pages) return []

      const sidebar = props.pages === '/' ? sidebarConfig.value : props.pages
      return processSidebar(sidebar)
    })

    // 统计信息（仅在顶层计算）
    // 列入目录的文章数（有链接且字数超过100的页面）
    const tocArticleCount = computed(() => {
      if (props.level !== 0 || props.pages !== '/') return 0

      const sidebar = sidebarConfig.value
      if (!sidebar || sidebar.length === 0) return 0

      let count = 0
      const countArticles = (items) => {
        for (const item of items) {
          if (item.link) {
            const wordCount = findWordCount(item.link)
            if (wordCount > 100) count++
          }
          if (item.children) countArticles(item.children)
        }
      }
      countArticles(sidebar)
      return count
    })

    // 合计总字数（只统计目录内的文章）
    const totalWords = computed(() => {
      if (props.level !== 0) return 0

      const sidebar = sidebarConfig.value
      if (!sidebar || sidebar.length === 0) return 0

      let total = 0
      const countWords = (items) => {
        for (const item of items) {
          if (item.link) {
            const wordCount = findWordCount(item.link)
            if (wordCount > 100) total += wordCount
          }
          if (item.children) countWords(item.children)
        }
      }
      countWords(sidebar)
      return total
    })

    // 最后更新日期（使用当前日期，因为无法在客户端获取git信息）
    const lastUpdateDate = computed(() => {
      if (props.level !== 0) return ''
      // 格式化当前日期
      const now = new Date()
      return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    })

    // 处理侧边栏配置
    function processSidebar(items) {
      if (!Array.isArray(items)) return []

      return items
        .filter(item => {
          // 过滤掉目录页本身（link 指向 SUMMARY）
          if (item.link && (item.link === 'SUMMARY' || item.link.includes('SUMMARY'))) {
            return false
          }
          return true
        })
        .map(item => {
        const result = {
          title: getTitle(item),
          words: getWords(item),
          links: getLinks(item),
          children: []
        }

        if (item.children && Array.isArray(item.children)) {
          result.children = processSidebar(item.children)
        }

        return result
      })
    }

    function getTitle(item) {
      if (typeof item === 'string') {
        return item
      }
      return item.text || item.title || '未知标题'
    }

    function getWords(item) {
      // 优先检查已处理的 words 属性
      if (item.words) {
        return item.words
      }

      if (typeof item === 'string') {
        const wordCount = findWordCount(item)
        return wordCount > 0 ? `${wordCount.toLocaleString()} 字` : ''
      }

      if (item.link) {
        const wordCount = findWordCount(item.link)
        return wordCount > 0 ? `${wordCount.toLocaleString()} 字` : ''
      }

      if (item.children && Array.isArray(item.children)) {
        const totalWords = item.children.reduce((sum, child) => {
          return sum + getWordCountNum(child)
        }, 0)
        return totalWords > 0 ? `${totalWords.toLocaleString()} 字` : ''
      }

      return ''
    }

    function getWordCountNum(item) {
      if (typeof item === 'string') {
        return findWordCount(item) || 0
      }

      if (item.link) {
        return findWordCount(item.link) || 0
      }

      // 递归处理有 children 的项目
      if (item.children && Array.isArray(item.children)) {
        return item.children.reduce((sum, child) => {
          return sum + getWordCountNum(child)
        }, 0)
      }

      return 0
    }

    function getLinks(item) {
      // 优先检查已处理的 links 属性，然后检查原始 link 属性
      if (item.links) {
        return item.links
      }

      if (typeof item === 'string') {
        const wordCount = findWordCount(item)
        return wordCount > 100 ? item : null
      }

      if (item.link) {
        const wordCount = findWordCount(item.link)
        return wordCount > 100 ? item.link : null
      }

      return null
    }

    function findWordCount(linkPath) {
      const data = wordCountData
      if (!data || Object.keys(data).length === 0) return 0

      // 标准化路径
      const normalizedPath = linkPath.startsWith('/') ? linkPath : '/' + linkPath

      // 尝试多种路径格式
      const candidates = [
        normalizedPath,
        normalizedPath + '.html',
        normalizedPath.replace(/\.html$/, ''),
        normalizedPath + '/',
      ]

      for (const candidate of candidates) {
        if (data[candidate] !== undefined) {
          // wordCountData 结构为 {title, wordCount}
          return data[candidate].wordCount || data[candidate]
        }
      }

      return 0
    }

    return {
      information,
      // 统计信息（仅在顶层显示）
      tocArticleCount,
      totalWords,
      lastUpdateDate
    }
  }
})
</script>

<style scoped>
.global-toc {
  padding: 0;
}

.stats-info {
  background: var(--c-tip-bg, #f3f5f7);
  border-radius: 4px;
  padding: 16px 20px;
  margin-bottom: 20px;
  border-left: 5px solid var(--c-tip-border, #2563EB);
}

.stats-info p {
  margin: 0;
  color: var(--c-text, #2c3e50);
  font-size: 14px;
  line-height: 1.6;
}

.stats-info strong {
  color: var(--c-brand, #2563EB);
  font-weight: 600;
}

ol {
  padding: 0;
  margin: 0;
  list-style: none;
}

/* 嵌套列表缩进 */
ol ol {
  margin-left: 24px;
}

li > span {
  display: block;
}

.words {
  font-size: 14px;
  color: #999;
  float: right;
  margin-right: 10px;
}

.level0 {
  font-size: 17px;
  line-height: 44px;
  font-weight: bold;
}

.level1 {
  font-size: 15px;
  line-height: 35px;
}

.level2 {
  font-size: 14px;
  line-height: 30px;
}

a {
  color: var(--c-text);
  text-decoration: none;
}

a:hover {
  color: var(--c-brand);
}
</style>