<template>
  <DefaultLayout>
    <!-- 导航栏右侧添加设置按钮 -->
    <template #navbar-after>
      <NavbarSettings />
    </template>
    <!-- 自定义页面顶部：显示 tags -->
    <template #page-content-top>
      <div v-if="pageTags.length" class="page-tags">
        <span v-for="tag in pageTags" :key="tag" class="tag">{{ tag }}</span>
      </div>
    </template>
    <!-- 自定义页面底部：文章元信息 -->
    <template #page-content-bottom>
      <ArticleFooter />
    </template>
    <!-- 页面最底部：评论（在翻页导航之后） -->
    <template #page-bottom>
      <Comments />
    </template>
  </DefaultLayout>
</template>

<script setup>
import { computed } from 'vue'
import { usePageFrontmatter } from '@vuepress/client'
import DefaultLayout from '@vuepress/theme-default/lib/client/layouts/Layout.vue'
import ArticleFooter from '../components/ArticleFooter.vue'
import NavbarSettings from '../components/NavbarSettings.vue'
import Comments from '../components/Comments.vue'

const frontmatter = usePageFrontmatter()
const pageTags = computed(() => frontmatter.value?.tags || [])
</script>

<style>
/* 自定义样式覆盖 */
.vp-theme-container {
  --vp-c-accent: #2563EB;
  --vp-c-accent-bg: #2563EB;
  --vp-c-accent-hover: #3B82F6;
  --vp-c-accent-soft: #EFF6FF;
}

/* Tags 样式 - 与 pencil 设计稿一致 */
.page-tags {
  padding: 0 64px;
  margin-bottom: 1rem;
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.page-tags .tag {
  display: inline-block;
  padding: 4px 12px;
  background: #EFF6FF;
  color: #2563EB;
  border-radius: 100px;
  font-size: 12px;
  font-weight: 500;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

@media (max-width: 959px) {
  .page-tags {
    padding: 0 48px;
  }
}

@media (max-width: 719px) {
  .page-tags {
    padding: 0 32px;
  }
}

@media (max-width: 419px) {
  .page-tags {
    padding: 0 16px;
  }
}
</style>