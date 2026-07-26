-- articles/word/filters/code-block.lua
-- 给代码块应用「代码清单」样式

function CodeBlock(el)
  -- 给代码块添加 custom-style 属性
  -- Pandoc DOCX writer 会将此属性映射为 Word 段落样式
  el.attributes['custom-style'] = '代码清单'
  return el
end

function Code(el)
  -- 内联代码也做标记（可选，Word 对 inline code 样式支持有限）
  return el
end
