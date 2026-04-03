<template>
  <div class="wh_slide" @click="clickSlide">
    <div class="wh_slide_content">
      <slot />
    </div>
  </div>
</template>

<script>
export default {
  name: 'Slide',
  mounted() {
    // 禁用轮播图片的点击放大功能，避免干扰滑动操作
    this.$el.querySelectorAll('img').forEach(img => {
      img.setAttribute('data-no-zoom', 'true')
    })
  },
  methods: {
    clickSlide() {
      this.$emit('click')
    }
  }
}
</script>

<style scoped>
.wh_slide {
  width: 100%;
  flex-shrink: 0;
  z-index: 10;
  min-height: 100px;
}

.wh_slide_content {
  display: flex;
  justify-content: center;
  align-items: center;
}

.wh_slide img {
  display: block;
  max-width: 100%;
  cursor: grab;
}

.wh_slide img:active {
  cursor: grabbing;
}
</style>