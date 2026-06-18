/**
 * Mermaid 瀹㈡埛绔厤缃?
 * 鍔ㄦ€佸姞杞?Mermaid 搴撳苟鍒濆鍖?
 */
import { defineClientConfig, usePageData } from 'vuepress/client'
import { onMounted, watch, ref } from 'vue'
import { useRouter } from 'vue-router'

// 鍏ㄥ眬 mermaid 瀹炰緥
let mermaidInstance = null
let mermaidLoaded = false

/**
 * 娓叉煋椤甸潰涓殑 mermaid 鍥捐〃
 */
async function renderMermaid() {
  const mermaidModule = await import('mermaid').catch(() => null)
  if (!mermaidModule) return

  const mermaid = mermaidModule.default
  mermaidInstance = mermaid

  // 鍒濆鍖?mermaid锛堝彧鎵ц涓€娆★級
  if (!mermaidLoaded) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        // 澶у箙鍑忓皬鑺傜偣闂磋窛鍜屽眰绾ч棿璺濓紝浣垮浘琛ㄦ洿绱у噾
        nodeSpacing: 20,
        rankSpacing: 25,
        // 浣跨敤璐濆灏旀洸绾匡紝杩炵嚎鏇寸揣鍑?
        curve: 'basis',
      },
      themeVariables: {
        // 榛樿瀛椾綋浠?14px 鍑忓皬鍒?11px锛岃妭鐐逛細鏇村皬
        fontSize: '11px',
        // 鑺傜偣鏍峰紡 - 涓?SVG 鍥剧墖涓€鑷?
        primaryColor: '#eeeeee',
        primaryBorderColor: '#999999',
        primaryTextColor: '#333333',
        // 鑺傜偣杈规瀹藉害浠?2px 鍑忓皬鍒?1px
        nodeBorderWidth: '1px',
        // 绾挎潯鏍峰紡
        lineColor: '#333333',
        // 绾挎潯瀹藉害鍙樼粏
        edgeLineWidth: '1px',
        // 鑿卞舰鑺傜偣鏍峰紡
        secondaryColor: '#eeeeee',
        secondaryBorderColor: '#999999',
        secondaryTextColor: '#333333',
        edgeLabelBackground: `#ffffff`,
        // 鍑忓皬鑺傜偣鍐呴儴 padding
        nodePadding: 8
      }
    })
    mermaidLoaded = true
    window.mermaid = mermaid
  }

  // 鏌ユ壘鎵€鏈夋湭娓叉煋鐨?mermaid pre 鍏冪礌
  const mermaidEls = document.querySelectorAll('pre.mermaid')

  for (const el of mermaidEls) {
    // 璺宠繃宸茬粡娓叉煋杩囩殑鍏冪礌
    if (el.dataset.rendered === 'true') continue

    // 妫€鏌ョ埗鑺傜偣鏄惁瀛樺湪锛堥伩鍏?Vue 鍝嶅簲寮忔洿鏂板鑷村厓绱犺绉婚櫎锛?
    if (!el.parentNode) {
      console.warn('Mermaid element has no parent node, skipping')
      continue
    }

    // 浠?data-mermaid 灞炴€ц幏鍙栧師濮嬩唬鐮侊紙閬垮厤娴忚鍣ㄨВ鏋?HTML 鏍囩锛?
    const code = el.dataset.mermaid ? decodeURIComponent(el.dataset.mermaid) : el.textContent

    // 鎻愬彇灏哄淇グ绗︾被鍚嶏紙mermaid-small, mermaid-compact, mermaid-tiny锛?
    const sizeClasses = el.className.split(' ').filter(cls => cls.startsWith('mermaid-') && cls !== 'mermaid')

    if (code && code.trim()) {
      try {
        // 鍏堟爣璁颁负宸叉覆鏌擄紝闃叉閲嶅澶勭悊
        el.dataset.rendered = 'true'

        // 鍒涘缓鏂扮殑 div 瀹瑰櫒锛屼繚鐣欏昂瀵镐慨楗扮
        const div = document.createElement('div')
        div.className = 'mermaid ' + sizeClasses.join(' ')
        div.style.textAlign = 'center'
        div.textContent = code.trim()
        div.dataset.rendered = 'true'

        // 鏇挎崲鍘熸潵鐨?pre 鍏冪礌
        el.parentNode.replaceChild(div, el)

        // 娓叉煋
        await mermaid.run({ nodes: [div] })

        // 娓叉煋鍚庣‘淇濆眳涓紝骞舵牴鎹缉鏀捐皟鏁村鍣ㄥぇ灏?
        const svg = div.querySelector('svg')
        if (svg) {
          svg.style.display = 'inline-block'
          svg.style.margin = '0 auto'

          // 淇 subgraph 鏍囬瑁佸壀锛歁ermaid 瀵?cluster label 鐨?<p> 鍏冪礌
          // 浣跨敤 16px 瀛椾綋锛堜笉鍙?themeVariables.fontSize 鎺у埗锛夛紝瀵艰嚧鏍囬楂樺害
          // 瓒呭嚭 foreignObject 鍜?cluster rect 鐨勯鐣欑┖闂达紝鏍囬涓婂崐閮ㄥ垎琚鍓€?
          // 淇鏂瑰紡锛氬浐瀹?<p> 瀛椾綋涓?16px銆乴ineHeight 1.5锛屽皢 foreignObject
          // 楂樺害璁句负 26px锛堣冻澶熷绾?16px*1.5=24px 琛岄珮鍔?2px 浣欓噺锛?
          const clusterLabels = svg.querySelectorAll('.cluster foreignObject p')
          for (const p of clusterLabels) {
            p.style.fontSize = '16px'
            p.style.lineHeight = '1.5'
            p.style.margin = '0'
            p.style.marginTop = '-3px'
          }
          const foreignObjects = svg.querySelectorAll('.cluster foreignObject')
          for (const fo of foreignObjects) {
            fo.setAttribute('height', '26')
          }

          // 濡傛灉鏈夊昂瀵镐慨楗扮锛岃皟鏁村鍣ㄥぇ灏忎互閬垮厤瑁佸壀
          const sizeClass = sizeClasses.find(cls => cls.startsWith('mermaid-'))
          if (sizeClass) {
            // 鑾峰彇缂╂斁姣斾緥
            const scales = {
              'mermaid-small': 0.85,
              'mermaid-compact': 0.75,
              'mermaid-tiny': 0.6
            }
            const scale = scales[sizeClass] || 1

            // 鑾峰彇 SVG 鐨勫師濮嬪昂瀵?
            const viewBox = svg.getAttribute('viewBox')
            if (viewBox) {
              const parts = viewBox.split(' ')
              const width = parseFloat(parts[2])
              const height = parseFloat(parts[3])

              // 璁剧疆瀹瑰櫒瀹介珮涓虹缉鏀惧悗鐨勫昂瀵?
              div.style.width = `${width * scale}px`
              div.style.height = `${height * scale}px`
              div.style.overflow = 'hidden'
              div.style.display = 'inline-block'
              div.style.textAlign = 'left'

              // SVG 浣跨敤 transform scale
              svg.style.transform = `scale(${scale})`
              svg.style.transformOrigin = 'top left'
              // SVG 淇濇寔鍘熷灏哄
              svg.style.width = `${width}px`
              svg.style.height = `${height}px`
            }
          }
        }
      } catch (err) {
        console.error('Mermaid render error:', err)
      }
    }
  }
}

export default defineClientConfig({
  enhance({ app }) {
    // 棰勫姞杞?mermaid 搴?
    if (typeof window !== 'undefined') {
      import('mermaid').then((mermaidModule) => {
        const mermaid = mermaidModule.default
        window.mermaid = mermaid

        mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
            // 澶у箙鍑忓皬鑺傜偣闂磋窛鍜屽眰绾ч棿璺濓紝浣垮浘琛ㄦ洿绱у噾
            nodeSpacing: 20,
            rankSpacing: 25,
            // 浣跨敤璐濆灏旀洸绾匡紝杩炵嚎鏇寸揣鍑?
            curve: 'basis',
            // subgraph 鏍囬杈硅窛锛屼负鏍囬棰勭暀绌洪棿閬垮厤琚鍓?
            subGraphTitleMargin: {
              top: 4,
              bottom: 9
            }
          },
          themeVariables: {
            // 榛樿瀛椾綋浠?14px 鍑忓皬鍒?11px锛岃妭鐐逛細鏇村皬
            fontSize: '11px',
            // 鑺傜偣鏍峰紡 - 涓?SVG 鍥剧墖涓€鑷?
            primaryColor: '#eeeeee',
            primaryBorderColor: '#999999',
            primaryTextColor: '#333333',
            // 鑺傜偣杈规瀹藉害浠?2px 鍑忓皬鍒?1px
            nodeBorderWidth: '1px',
            // 绾挎潯鏍峰紡
            lineColor: '#333333',
            // 绾挎潯瀹藉害鍙樼粏
            edgeLineWidth: '1px',
            // 鑿卞舰鑺傜偣鏍峰紡
            secondaryColor: '#eeeeee',
            secondaryBorderColor: '#999999',
            secondaryTextColor: '#333333',
            edgeLabelBackground: `#ffffff`,
            // 鍑忓皬鑺傜偣鍐呴儴 padding
            nodePadding: 8
          }
        })
        mermaidLoaded = true
        mermaidInstance = mermaid
      }).catch(err => {
        console.error('Failed to load mermaid:', err)
      })
    }
  },

  setup() {
    // 浣跨敤 Vue Router 鐩戝惉璺敱鍙樺寲
    const router = useRouter()
    const isInitialized = ref(false)
    const pageData = usePageData()

    // 缁勪欢鎸傝浇鏃舵覆鏌?
    onMounted(() => {
      // 寤惰繜娓叉煋锛岀‘淇?DOM 宸插姞杞?
      setTimeout(renderMermaid, 100)
      setTimeout(renderMermaid, 300)
    })

    // 鐩戝惉璺敱鍙樺寲锛屾瘡娆¤矾鐢卞垏鎹㈠悗閲嶆柊娓叉煋
    watch(
      () => router.currentRoute.value.path,
      (newPath, oldPath) => {
        // 璺敱鍙樺寲鍚庡欢杩熸覆鏌?
        if (oldPath !== undefined) {
          setTimeout(renderMermaid, 100)
          setTimeout(renderMermaid, 300)
          setTimeout(renderMermaid, 500)
        }
      },
      { immediate: false }
    )

    // 鐩戝惉椤甸潰鏁版嵁鍙樺寲锛堢儹鏇存柊浼氳Е鍙戞鍙樺寲锛夛紝閲嶆柊娓叉煋 mermaid 鍥捐〃
    watch(
      () => pageData.value,
      () => {
        setTimeout(renderMermaid, 100)
        setTimeout(renderMermaid, 300)
      },
      { deep: true }
    )
  }
})



