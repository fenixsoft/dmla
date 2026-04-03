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
}
</style>