<template>
  <span class="github-button-wrapper">
    <a
      ref="buttonLink"
      :href="href"
      :aria-label="ariaLabel"
      :data-icon="dataIcon"
      :data-show-count="dataShowCount"
      :data-text="dataText"
    >
      <slot>Star</slot>
    </a>
  </span>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, onBeforeUpdate } from 'vue'

const props = defineProps({
  href: {
    type: String,
    required: true
  },
  ariaLabel: {
    type: String,
    default: ''
  },
  dataIcon: {
    type: String,
    default: 'octicon-star'
  },
  dataShowCount: {
    type: Boolean,
    default: true
  },
  dataText: {
    type: String,
    default: ''
  }
})

const buttonLink = ref(null)
let renderedElement = null

const renderButton = async () => {
  if (!buttonLink.value) return

  try {
    const module = await import('github-buttons')
    module.render(buttonLink.value, (el) => {
      if (buttonLink.value && buttonLink.value.parentNode) {
        renderedElement = el
        buttonLink.value.parentNode.replaceChild(el, buttonLink.value)
      }
    })
  } catch (e) {
    // github-buttons 加载失败时，保留原始链接作为 fallback
    console.warn('GitHub buttons 加载失败，使用静态链接:', e)
  }
}

const resetButton = () => {
  if (renderedElement && renderedElement.parentNode) {
    renderedElement.parentNode.replaceChild(buttonLink.value, renderedElement)
    renderedElement = null
  }
}

onMounted(() => {
  renderButton()
})

onBeforeUpdate(() => {
  resetButton()
})

onBeforeUnmount(() => {
  resetButton()
})
</script>

<style scoped>
.github-button-wrapper {
  display: inline-flex;
  align-items: center;
}
</style>