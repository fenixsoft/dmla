<script setup>
import { computed } from 'vue'
import { usePageFrontmatter, useSiteLocaleData, withBase, AutoLink, ClientOnly } from 'vuepress/client'

const frontmatter = usePageFrontmatter()
const siteLocale = useSiteLocaleData()

// 从 frontmatter 或站点配置获取数据
const heroText = computed(() => {
  if (frontmatter.value.heroText === null) return null
  return frontmatter.value.heroText || siteLocale.value.title || 'Hello'
})

const tagline = computed(() => {
  if (frontmatter.value.tagline === null) return null
  return frontmatter.value.tagline || siteLocale.value.description || 'Welcome'
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

    <!-- 描述 -->
    <p v-if="tagline" class="home-hero-tagline">
      {{ tagline }}
    </p>

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
  max-height: 280px;
  margin: 3rem auto 1.5rem;
}

.home-hero-title {
  font-size: 3rem;
  margin: 1.8rem auto;
}

.home-hero-tagline {
  max-width: 35rem;
  margin: 1.8rem auto;
  color: var(--vp-c-text-mute);
  font-size: 1.6rem;
  line-height: 1.3;
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
  color: var(--vp-c-accent-text);
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