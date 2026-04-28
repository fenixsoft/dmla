<template>
  <div class="global-toc">
    <!-- 统计信息 -->
    <div v-if="level === 0" class="stats-info">
      <p>共计 <strong>{{ tocArticleCount }}</strong> 篇文章，合计 <strong :title="totalWordsHint">{{ totalWords.toLocaleString() }}</strong> 字，最后更新日期 <strong>{{ lastUpdateDate }}</strong>。</p>
    </div>
    <ol>
      <li v-for="(page, index) in information" :key="index">
        <span v-if="page.links != null">
          <a :href="page.links">
            <span :class="'level' + level">{{ page.title }}</span>
          </a>
          <span class="words" :title="page.wordHint">{{ page.words }}</span>
        </span>
        <span v-else :class="'level' + level">
          {{ page.title }}
          <span class="words" :title="page.wordHint">{{ page.words }}</span>
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

    // 合计分类字数（文字和代码）
    const totalTextWords = computed(() => {
      if (props.level !== 0) return 0

      const sidebar = sidebarConfig.value
      if (!sidebar || sidebar.length === 0) return 0

      let total = 0
      const countTextWords = (items) => {
        for (const item of items) {
          if (item.link) {
            const data = findWordCountData(item.link)
            if (data.wordCount > 100) total += data.textWordCount || 0
          }
          if (item.children) countTextWords(item.children)
        }
      }
      countTextWords(sidebar)
      return total
    })

    const totalCodeWords = computed(() => {
      if (props.level !== 0) return 0

      const sidebar = sidebarConfig.value
      if (!sidebar || sidebar.length === 0) return 0

      let total = 0
      const countCodeWords = (items) => {
        for (const item of items) {
          if (item.link) {
            const data = findWordCountData(item.link)
            if (data.wordCount > 100) total += data.codeWordCount || 0
          }
          if (item.children) countCodeWords(item.children)
        }
      }
      countCodeWords(sidebar)
      return total
    })

    // 合计字数的悬浮提示
    const totalWordsHint = computed(() => {
      if (props.level !== 0) return ''
      return `文字：${totalTextWords.value.toLocaleString()} 字\n代码：${totalCodeWords.value.toLocaleString()} 字`
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
          // 过滤掉目录页本身（link 指向 contents）
          if (item.link && (item.link === 'contents' || item.link.includes('contents'))) {
            return false
          }
          return true
        })
        .map(item => {
        const wordData = getWordData(item)
        const result = {
          title: getTitle(item),
          words: wordData.display,
          wordHint: wordData.hint,
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

    function getWordData(item) {
      // 返回 {display: '字数显示', hint: '悬浮提示'}
      // 优先检查已处理的 words 和 wordHint 属性（用于递归渲染时）
      if (item.words && item.wordHint) {
        return { display: item.words, hint: item.wordHint }
      }

      if (typeof item === 'string') {
        const data = findWordCountData(item)
        return formatWordData(data)
      }

      // 检查 link 或 links 属性（links 是处理后的属性）
      const linkPath = item.link || item.links
      if (linkPath) {
        const data = findWordCountData(linkPath)
        return formatWordData(data)
      }

      if (item.children && Array.isArray(item.children)) {
        // 递归合计所有后代页面的字数
        let totalText = 0, totalCode = 0
        const sumChildrenWords = (children) => {
          for (const child of children) {
            // child 可能是字符串、原始 sidebar item，或已处理的 result
            let childLink = typeof child === 'string' ? child : (child.link || child.links || '')
            if (childLink) {
              const childData = findWordCountData(childLink)
              totalText += childData.textWordCount || 0
              totalCode += childData.codeWordCount || 0
            }
            // 如果子项还有 children，递归统计
            if (child.children && Array.isArray(child.children)) {
              sumChildrenWords(child.children)
            }
          }
        }
        sumChildrenWords(item.children)
        return formatWordData({ wordCount: totalText + totalCode, textWordCount: totalText, codeWordCount: totalCode })
      }

      return { display: '', hint: '' }
    }

    function formatWordData(data) {
      const wordCount = data.wordCount || 0
      const textWordCount = data.textWordCount || 0
      const codeWordCount = data.codeWordCount || 0
      if (wordCount <= 0) return { display: '', hint: '' }
      return {
        display: `${wordCount.toLocaleString()} 字`,
        hint: `文字：${textWordCount.toLocaleString()} 字\n代码：${codeWordCount.toLocaleString()} 字`
      }
    }

    function getLinks(item) {
      // 优先检查已处理的 links 属性
      if (item.links) {
        return item.links
      }

      if (typeof item === 'string') {
        const data = findWordCountData(item)
        return data.wordCount > 100 ? item : null
      }

      // 检查原始的 link 属性
      if (item.link) {
        const data = findWordCountData(item.link)
        return data.wordCount > 100 ? item.link : null
      }

      return null
    }

    function findWordCountData(linkPath) {
      // 返回 {wordCount, textWordCount, codeWordCount}
      const data = wordCountData
      if (!data || Object.keys(data).length === 0) return { wordCount: 0, textWordCount: 0, codeWordCount: 0 }

      // 处理输入可能是对象的情况
      let path = linkPath
      if (typeof linkPath === 'object' && linkPath !== null) {
        path = linkPath.link || ''
      }

      if (typeof path !== 'string' || !path) {
        return { wordCount: 0, textWordCount: 0, codeWordCount: 0 }
      }

      // 标准化路径
      const normalizedPath = path.startsWith('/') ? path : '/' + path

      // 尝试多种路径格式
      const candidates = [
        normalizedPath,
        normalizedPath + '.html',
        normalizedPath.replace(/\.html$/, ''),
        normalizedPath + '/',
      ]

      for (const candidate of candidates) {
        if (data[candidate] !== undefined) {
          // wordCountData 结构为 {title, wordCount, textWordCount, codeWordCount}
          const entry = data[candidate]
          return {
            wordCount: entry.wordCount || 0,
            textWordCount: entry.textWordCount || 0,
            codeWordCount: entry.codeWordCount || 0
          }
        }
      }

      return { wordCount: 0, textWordCount: 0, codeWordCount: 0 }
    }

    // 兼容旧的 findWordCount 函数（用于统计信息）
    function findWordCount(linkPath) {
      return findWordCountData(linkPath).wordCount
    }

    return {
      information,
      // 统计信息（仅在顶层显示）
      tocArticleCount,
      totalWords,
      totalTextWords,
      totalCodeWords,
      totalWordsHint,
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