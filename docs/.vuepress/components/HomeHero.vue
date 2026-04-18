<script setup>
import { computed } from 'vue'
import { usePageFrontmatter, useSiteLocaleData, withBase, AutoLink } from 'vuepress/client'

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

  // 支持数组格式：[{ src, alt }]
  if (Array.isArray(rawInfo)) {
    return rawInfo.map(info => ({
      src: info.src,
      alt: info.alt || ''
    }))
  }

  return []
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

    <!-- Informations -->
    <div v-if="informations.length" class="home-hero-informations">
      <img
        v-for="info in informations"
        :key="info.src"
        :src="info.src"
        :alt="info.alt"
      />
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
  /* height: 20px; */
  vertical-align: middle;
  margin: 0px;
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