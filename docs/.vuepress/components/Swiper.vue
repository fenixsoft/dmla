<template>
  <section class="wh_content" @touchmove="fn">
    <div :class="className" class="wh_swiper" @touchstart="s" @touchmove="m" @touchend="e" @click="nextSlide">
      <slot />
    </div>

    <div v-if="showIndicator" class="wh_indicator">
      <div
        v-for="(tag, $index) in slidesLength"
        :key="$index"
        :class="{ wh_show_bgcolor: index - 1 === $index }"
        class="wh_indicator_item"
        @click="slideTo($index)"
      />
    </div>
  </section>
</template>

<script>
export default {
  name: 'Swiper',
  data() {
    return {
      slidesLength: 1,
      _width: 0,
      auto: true,
      slideing: true,
      timer1: '',
      className: '',
      dom: {},
      t: {
        sx: 0,
        s: 0,
        m: 0,
        e: 0
      },
      index: 1
    }
  },
  props: {
    autoPlay: {
      type: Boolean,
      default: true
    },
    duration: {
      type: Number,
      default: 500
    },
    interval: {
      type: Number,
      default: 5000
    },
    showIndicator: {
      type: Boolean,
      default: true
    }
  },
  mounted() {
    this.className = `wh_swiper_${Math.floor(Math.random() * 1000)}`
    // 使用 nextTick 确保 DOM 更新完成，然后再延迟一点确保样式计算完成
    this.$nextTick(() => {
      setTimeout(() => {
        this.starDom()
        if (this.slidesLength > 1) {
          this.dom.transform = `translate3d(${this._width * -1}px, 0px, 0px)`
          this.dom['-webkit-transform'] = `translate3d(${this._width * -1}px, 0px, 0px)`
          if (this.autoPlay) {
            this.setTime()
          }
        }
      }, 100)
    })

    window.addEventListener('resize', this.handleResize)
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.handleResize)
    window.clearTimeout(this.timer1)
  },
  methods: {
    handleResize() {
      if (this.slidesLength > 1) {
        this._width = document.querySelector('.' + this.className).offsetWidth
        this.dom.transform = `translate3d(${this._width * -1 * this.index}px, 0px, 0px)`
        this.dom['-webkit-transform'] = `translate3d(${this._width * -1 * this.index}px, 0px, 0px)`
      }
    },
    s(x) {
      if (this.slideing && this.slidesLength > 1) {
        this.clearTimeOut()
        this.t.sx = this.getTransform()
        this.t.s = x.touches[x.touches.length - 1].clientX
      }
    },
    m(x) {
      if (this.slideing && this.t.s !== -1 && this.slidesLength > 1) {
        this.clearTimeOut()
        this.t.m = x.touches[x.touches.length - 1].clientX - this.t.s
        this.setTransform(this.t.m + this.t.sx)
      }
    },
    e() {
      if (this.slideing && this.t.s !== -1 && this.slidesLength > 1) {
        this.clearTimeOut()
        this.setTransform(this.t.m + this.t.sx)
        const x = this.getTransform()
        const threshold = this.t.m > 0 ? this._width * 0.3 : this._width * -0.3
        this.index = Math.round((x + threshold) / this._width) * -1
        this.wh('touch')
      }
    },
    setTransform(num) {
      this.dom.transform = `translate3d(${num}px, 0px, 0px)`
      this.dom['-webkit-transform'] = `translate3d(${num}px, 0px, 0px)`
    },
    getTransform() {
      const x = this.dom.transform || this.dom['-webkit-transform']
      if (!x) return 0
      const match = x.match(/translate3d\((\S+)px/)
      return match ? Number(match[1]) : 0
    },
    fn(e) {
      e.preventDefault()
    },
    prevSlide() {
      if (this.slidesLength > 1) {
        this.clearTimeOut()
        this.index--
        this.wh()
      }
    },
    nextSlide() {
      if (this.slidesLength > 1) {
        this.clearTimeOut()
        this.index++
        this.wh()
      }
    },
    slideTo(idx) {
      if (this.slidesLength > 1) {
        this.clearTimeOut()
        this.index = idx + 1
        this.wh()
      }
    },
    wh(type) {
      this.slideing = false
      this.dom.transition = type === 'touch' ? '250ms' : `${this.duration}ms`
      this.setTransform(this.index * -1 * this._width)
      this.t.m = 0
      this.t.s = -1
      if (this.autoPlay) {
        this.setTime()
      }
      const timeDuration = type === 'touch' ? 250 : this.duration
      setTimeout(() => {
        this.dom.transition = '0s'
        if (this.index >= this.slidesLength + 1) {
          this.index = 1
          this.setTransform(this.index * -1 * this._width)
        }
        if (this.index <= 0) {
          this.index = this.slidesLength
          this.setTransform(this.index * -1 * this._width)
        }
        this.$emit('transtionend', this.index - 1)
        this.auto = true
        this.slideing = true
      }, timeDuration)
    },
    setTime() {
      this.timer1 = window.setTimeout(() => {
        if (this.auto) {
          this.index++
          this.wh()
        } else {
          window.clearTimeout(this.timer1)
        }
      }, this.interval)
    },
    starDom() {
      const SlideDom = document.querySelector('.' + this.className)?.getElementsByClassName('wh_slide')
      if (!SlideDom || SlideDom.length === 0) return

      this.slidesLength = SlideDom.length
      if (this.slidesLength > 1) {
        const cloneDom1 = SlideDom[0].cloneNode(true)
        const cloneDom2 = SlideDom[this.slidesLength - 1].cloneNode(true)
        const container = document.querySelector('.' + this.className)
        container.insertBefore(cloneDom2, SlideDom[0])
        container.appendChild(cloneDom1)
        this._width = container.offsetWidth
        this.dom = container.style
      }
    },
    clearTimeOut() {
      this.auto = false
      window.clearTimeout(this.timer1)
    }
  }
}
</script>

<style>
.wh_content {
  position: relative;
  z-index: 1;
  overflow: hidden;
  width: 100%;
}

.wh_swiper {
  width: 100%;
  display: flex;
  transition-duration: 0s linear;
}

.wh_indicator {
  position: absolute;
  bottom: 8px;
  width: 100%;
  text-align: center;
  background: transparent;
}

.wh_indicator_item {
  display: inline-block;
  width: 12px;
  height: 12px;
  margin: 1px 7px;
  cursor: pointer;
  border-radius: 100%;
  background: #aaa;
}

.wh_show_bgcolor {
  background: #3884fe;
}
</style>