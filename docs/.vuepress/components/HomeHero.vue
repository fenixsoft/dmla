<script setup>
import { ref, computed, onMounted } from 'vue'
import { usePageFrontmatter, useSiteLocaleData, withBase, AutoLink } from 'vuepress/client'
import GithubInfos from './GithubInfos.vue'

const frontmatter = usePageFrontmatter()
const siteLocale = useSiteLocaleData()

// 从 frontmatter 或站点配置获取数据
const heroText = computed(() => {
  if (frontmatter.value.heroText === null) return null
  return frontmatter.value.heroText || siteLocale.value.title || 'Hello'
})

const heroImage = computed(() => frontmatter.value.heroImage)
const heroAlt = computed(() => frontmatter.value.heroAlt || heroText.value || 'hero')

const actions = computed(() => {
  if (!Array.isArray(frontmatter.value.actions)) return []
  return frontmatter.value.actions.map(({ text, link, type = 'primary' }) => ({
    text,
    link,
    type
  }))
})

const features = computed(() => frontmatter.value.features ?? [])

// 解析 badges 数据
const badges = computed(() => {
  const rawBadges = frontmatter.value.badges
  if (!rawBadges) return []

  // 支持数组格式：[{ src, alt, href }]
  if (Array.isArray(rawBadges)) {
    return rawBadges.map(badge => ({
      src: badge.src,
      alt: badge.alt || '',
      href: badge.href || ''
    }))
  }

  return []
})

// 解析 informations 数据
const informations = computed(() => {
  const rawInfo = frontmatter.value.informations
  if (!rawInfo) return []
  if (Array.isArray(rawInfo)) return rawInfo
  return []
})

// XML 转义：防止 SVG 中的特殊字符导致渲染异常
function escapeXml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

// Canvas 测量文本宽度（懒初始化，SSR 环境回退到估算值）
let _measureCtx = null
function measureTextWidth(text) {
  if (typeof document === 'undefined') {
    // SSR 环境回退：每字符约 6.5px
    return String(text).length * 6.5
  }
  if (!_measureCtx) {
    const canvas = document.createElement('canvas')
    _measureCtx = canvas.getContext('2d')
    _measureCtx.font = '11px DejaVu Sans, Verdana, Geneva, sans-serif'
  }
  return _measureCtx.measureText(String(text)).width
}

// 生成 shields.io 风格的 SVG badge 数据 URI
// 图片渲染不受宿主页面 CSS 影响，确保所有设备上字体大小和间距完全一致
function makeBadgeSvg(label, value, color) {
  const escapedLabel = escapeXml(String(label))
  const escapedValue = escapeXml(String(value))
  const fontFamily = 'DejaVu Sans,Verdana,Geneva,sans-serif'
  const fontSize = 11
  const paddingX = 6
  const height = 20

  const labelTextW = measureTextWidth(label)
  const valueTextW = measureTextWidth(value)
  const labelW = Math.ceil(labelTextW + paddingX * 2)
  const valueW = Math.ceil(valueTextW + paddingX * 2)
  const totalW = labelW + valueW

  const labelCx = labelW / 2
  const valueCx = labelW + valueW / 2

  // 生成与 shields.io 结构一致的 SVG：左右分区、渐变阴影、文字投影
  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${height}">`,
    `<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>`,
    `<clipPath id="r"><rect width="${totalW}" height="${height}" rx="3" fill="#fff"/></clipPath>`,
    `<g clip-path="url(#r)">`,
    `<rect width="${labelW}" height="${height}" fill="#555"/>`,
    `<rect x="${labelW}" width="${valueW}" height="${height}" fill="#${color}"/>`,
    `<rect width="${totalW}" height="${height}" fill="url(#s)"/>`,
    `</g>`,
    `<g fill="#fff" text-anchor="middle" font-family="${fontFamily}" font-size="${fontSize}">`,
    `<text x="${labelCx}" y="15" fill="#010101" opacity=".3">${escapedLabel}</text>`,
    `<text x="${labelCx}" y="14">${escapedLabel}</text>`,
    `<text x="${valueCx}" y="15" fill="#010101" opacity=".3">${escapedValue}</text>`,
    `<text x="${valueCx}" y="14">${escapedValue}</text>`,
    `</g>`,
    `</svg>`
  ].join('')

  return 'data:image/svg+xml,' + encodeURIComponent(svg)
}

// API 驱动徽章的值缓存
const apiValues = ref({})

// 计算每个 information badge 对应的 SVG 图片 URI
// static/github-api/npm-api 类型生成 SVG，url/github-badge 类型保持外部图片不用处理
const informationBadgeSvgs = computed(() => {
  return informations.value.map((info, idx) => {
    if (info.type === 'static') {
      return makeBadgeSvg(info.label, info.value, info.color || '555')
    }
    if (info.type === 'github-api' || info.type === 'npm-api') {
      return makeBadgeSvg(info.label, apiValues.value[idx] || '--', info.color || '555')
    }
    return null
  })
})

onMounted(async () => {
  for (const [idx, info] of informations.value.entries()) {
    if (info.type === 'github-api') {
      try {
        const res = await fetch(info.endpoint)
        if (res.ok) {
          const data = await res.json()
          let value = data[info.field]
          if (info.format === 'date' && value) {
            value = new Date(value).toISOString().slice(0, 10)
          }
          apiValues.value[idx] = value || '--'
        }
      } catch { apiValues.value[idx] = '--' }
    } else if (info.type === 'npm-api') {
      try {
        const res = await fetch(`https://registry.npmjs.org/${encodeURIComponent(info.package)}/latest`)
        if (res.ok) {
          const data = await res.json()
          apiValues.value[idx] = 'v' + data.version
        }
      } catch { apiValues.value[idx] = '--' }
    }
  }
})

// 解析 githubInfos 数据
const githubInfos = computed(() => {
  const raw = frontmatter.value.githubInfos
  if (!raw) return null

  // 支持布尔值 githubInfos: true，使用默认仓库
  if (raw === true) {
    return { repo: 'fenixsoft/dmla', owner: 'fenixsoft' }
  }

  // 支持对象格式 githubInfos: { repo: '...', owner: '...' }
  if (typeof raw === 'object') {
    return {
      repo: raw.repo || 'fenixsoft/dmla',
      owner: raw.owner || 'fenixsoft'
    }
  }

  return null
})
</script>

<template>
  <header class="home-hero">
    <!-- Hero 图片 -->
    <img
      v-if="heroImage"
      class="home-hero-image"
      :src="withBase(heroImage)"
      :alt="heroAlt"
    />

    <!-- 标题 -->
    <h1 v-if="heroText" class="home-hero-title">
      {{ heroText }}
    </h1>

    <!-- Badges -->
    <div v-if="badges.length" class="home-hero-badges">
      <a
        v-for="badge in badges"
        :key="badge.src"
        :href="badge.href"
        :target="badge.href ? '_blank' : undefined"
        :rel="badge.href ? 'noopener noreferrer' : undefined"
      >
        <img :src="badge.src" :alt="badge.alt" />
      </a>
    </div>

    <!-- GitHub 按钮（替代 shields.io 徽章） -->
    <div v-if="githubInfos" class="home-hero-github">
      <GithubInfos
        :repo="githubInfos.repo"
        :owner="githubInfos.owner"
      />
    </div>

    <!-- Informations 徽章（多类型：static / github-badge / github-api / npm-api / url） -->
    <div v-if="informations.length" class="home-hero-informations">
      <template v-for="(info, idx) in informations" :key="idx">
        <!-- 静态徽章：渲染为 SVG 图片，确保跨设备字体大小一致 -->
        <img v-if="info.type === 'static'" :src="informationBadgeSvgs[idx]" :alt="info.label + ': ' + info.value" />
        <!-- GitHub 原生图片徽章（CI 等） -->
        <img v-else-if="info.type === 'github-badge'" :src="info.src" :alt="info.alt" />
        <!-- API 驱动徽章（GitHub API / npm API）：渲染为 SVG 图片 -->
        <img v-else-if="info.type === 'github-api' || info.type === 'npm-api'" :src="informationBadgeSvgs[idx]" :alt="info.label" />
        <!-- 向后兼容：纯 URL 图片徽章 -->
        <img v-else :src="info.src" :alt="info.alt" />
      </template>
    </div>

    <!-- 操作按钮 -->
    <p v-if="actions.length" class="home-hero-actions">
      <AutoLink
        v-for="action in actions"
        :key="action.text"
        class="home-hero-action"
        :class="[action.type]"
        :config="action"
      />
    </p>
  </header>

  <!-- Features 区域 -->
  <div v-if="features.length" class="home-features">
    <div v-for="feature in features" :key="feature.title" class="home-feature">
      <h2>{{ feature.title }}</h2>
      <p>{{ feature.details }}</p>
    </div>
  </div>
</template>

<style scoped>
.home-hero {
  text-align: center;
  padding: 2rem 0;
}

.home-hero-image {
  display: block;
  max-width: 100%;
  max-height: 300px;
  margin: 3rem auto 1.5rem;
  box-shadow: none;
}

.home-hero-title {
  font-size: 3rem;
  padding-top: 0px;
  margin: -10px 0 25px 0;
}

.home-hero-tagline {
  max-width: 35rem;
  margin: 1.8rem auto;
  color: var(--vp-c-text-mute);
  font-size: 1.6rem;
  line-height: 1.3;
}

.home-hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin-bottom: -15px;
  color:chocolate
}

.home-hero-badges a::after {
  display: none !important;
}

.home-hero-github {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin: 0.5rem auto;
}

.home-hero-informations .medium-zoom-image {
  cursor: default;
}

.home-hero-badges img {
  height: 20px;
  vertical-align: middle;
}

.home-hero-informations {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin: 0.5rem auto;
}

.home-hero-informations img {
  vertical-align: middle;
  margin: 0px;
  cursor: default;
}

.home-hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: center;
  margin: 1.8rem auto;
}

.home-hero-action {
  display: inline-block;
  box-sizing: border-box;
  padding: 0.8rem 1.6rem;
  border: 2px solid var(--vp-c-accent-bg);
  border-radius: 4px;
  background-color: var(--vp-c-bg);
  color: var(--vp-c-accent);
  font-size: 1.2rem;
  transition: background-color border-color color 0.3s ease;
}

.home-hero-action:hover {
  color: var(-vp-c-accent);
  background-color: #eff4ff;
  text-decoration: none !important;
}

.home-hero-action.primary {
  background-color: var(--vp-c-accent-bg);
  color: var(--vp-c-accent-text);
}

.home-hero-action.primary:hover {
  border-color: var(--vp-c-accent-hover);
  background-color: var(--vp-c-accent-hover);
}

/* Features */
.home-features {
  display: flex;
  flex-wrap: wrap;
  place-content: stretch space-between;
  align-items: flex-start;
  margin-top: 2.5rem;
  padding: 1.2rem 0;
  border-top: 1px solid var(--vp-c-gutter);
}

.home-feature {
  flex-grow: 1;
  flex-basis: 30%;
  max-width: 30%;
}

.home-feature h2 {
  padding-bottom: 0;
  border-bottom: none;
  font-weight: 500;
  font-size: 1.4rem;
}

.home-feature p {
  color: var(--vp-c-text-mute);
}

/* 响应式 */
@media (max-width: 719px) {
  .home-features {
    flex-direction: column;
  }

  .home-feature {
    max-width: 100%;
    padding: 0 2.5rem;
  }
}

@media (max-width: 419px) {
  .home-hero-image {
    max-height: 210px;
    margin: 2rem auto 1.2rem;
  }

  .home-hero-title {
    font-size: 2rem;
    margin: 1.2rem auto;
  }

  .home-hero-tagline {
    font-size: 1.2rem;
    margin: 1.2rem auto;
  }

  .home-hero-actions {
    margin: 1.2rem auto;
  }

  .home-hero-action {
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
  }

  .home-feature h2 {
    font-size: 1.25rem;
  }
}
</style>