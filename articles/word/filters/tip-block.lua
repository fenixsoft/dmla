-- articles/word/filters/tip-block.lua
-- 将 Pandoc fenced div（::: {.tip} 等）转为带「注意」样式的段落

local CONTAINER_MAP = {
  tip = '注意',
  warning = '注意',
  danger = '注意',
  info = '注意',
  note = '注意',
  details = '注意',
}

function Div(el)
  -- 检查 fenced div 的 class
  for _, cls in ipairs(el.classes) do
    local style_name = CONTAINER_MAP[cls]
    if style_name then
      -- 创建一个带 custom-style 的 Div 包裹内容
      -- Pandoc 在 DOCX 输出时识别 custom-style 属性
      local new_div = pandoc.Div(el.content)
      new_div.attributes['custom-style'] = style_name
      return new_div
    end
  end
  return nil -- 不修改其他 Div
end
