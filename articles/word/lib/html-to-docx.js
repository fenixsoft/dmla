// articles/word/lib/html-to-docx.js
// 使用 Pandoc 将 HTML 内容转为 DOCX
import { execFileSync } from 'child_process';
import { existsSync, writeFileSync, mkdirSync, unlinkSync } from 'fs';
import { resolve } from 'path';

const TMP_DIR = resolve(import.meta.dirname, '../tmp');
const REF_DOCX = resolve(import.meta.dirname, '../reference.docx');
const FILTER_DIR = resolve(import.meta.dirname, '../filters');

/**
 * 检查 pandoc 命令是否可用
 */
function checkPandoc() {
  try {
    execFileSync('pandoc', ['--version'], { stdio: 'pipe', timeout: 5000 });
  } catch {
    throw new Error('Pandoc 未安装或不可用，请先运行 npm run setup');
  }
}

/**
 * 将 HTML 内容转为 DOCX 文件
 * @param {string} htmlContent - HTML 内容字符串
 * @param {string} title - 文章标题
 * @param {string} docxPath - 输出 DOCX 路径
 */
export function convertHtmlToDocx(htmlContent, title, docxPath) {
  checkPandoc();

  if (!existsSync(REF_DOCX)) {
    throw new Error(`参考模板不存在: ${REF_DOCX}`);
  }

  // 确保临时目录存在
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }

  // 包装为完整 HTML 文档，便于 Pandoc 解析
  const fullHtml = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<style>
  pre { background: #f5f5f5; padding: 1em; overflow-x: auto; font-family: Consolas, Monaco, monospace; font-size: 0.9em; line-height: 1.5; }
  code { font-family: Consolas, Monaco, monospace; font-size: 0.9em; background: #f0f0f0; padding: 0.1em 0.3em; }
  table { border-collapse: collapse; width: 100%; }
  table th, table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  table th { background: #f5f5f5; font-weight: bold; }
  img { max-width: 100%; height: auto; }
  blockquote { border-left: 4px solid #ddd; margin: 1em 0; padding: 0.5em 1em; color: #666; }
  .custom-container { border-left: 4px solid; padding: 0.5em 1em; margin: 1em 0; }
  .custom-container.tip { border-color: #42b983; background: #f0faf5; }
  .custom-container.warning { border-color: #e7c000; background: #fffdf0; }
  .custom-container.danger { border-color: #c00; background: #fff0f0; }
  .custom-container.info { border-color: #0070f3; background: #f0f7ff; }
  .custom-container.details { border-color: #888; background: #f8f8f8; }
</style>
</head>
<body>
${htmlContent}
</body>
</html>`;

  // 写入临时 HTML 文件
  const tmpHtmlPath = resolve(TMP_DIR, `${sanitizeFileName(title)}.html`);
  writeFileSync(tmpHtmlPath, fullHtml, 'utf-8');

  // Pandoc HTML → DOCX
  // 使用 --from=html 让 Pandoc 解析 HTML 输入
  const pandocArgs = [
    tmpHtmlPath,
    '-o', docxPath,
    '--from=html',
    '--metadata', `title=${escapeHtml(title)}`,
    `--reference-doc=${REF_DOCX}`,
    `--lua-filter=${resolve(FILTER_DIR, 'link-footnote.lua')}`,
    `--lua-filter=${resolve(FILTER_DIR, 'container-style.lua')}`,
    '--wrap=none',
  ];

  execFileSync('pandoc', pandocArgs, { stdio: 'pipe', timeout: 60000 });

  if (!existsSync(docxPath)) {
    throw new Error(`Pandoc 转换失败，未生成输出文件: ${docxPath}`);
  }

  // 后处理：修复字体和格式
  postProcessDocx(docxPath);
}

/**
 * 后处理 DOCX：
 * 1. 移除 Pandoc 写入的空 <w:rFonts>（覆盖样式字体）
 */
function postProcessDocx(docxPath) {
  const fixScript = docxPath.replace('.docx', '_fix.py');
  writeFileSync(fixScript, `
import zipfile, os
from lxml import etree

path = '${docxPath}'
ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
wp = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'

with zipfile.ZipFile(path, 'r') as z:
    files = {name: z.read(name) for name in z.namelist()}

# === Phase 1: Fix styles.xml (Theme attrs, footnote spacing) ===
if 'word/styles.xml' in files:
    tree = etree.fromstring(files['word/styles.xml'])
    for rFonts in tree.iter(f'{{{ns}}}rFonts'):
        for k in list(rFonts.attrib.keys()):
            if 'Theme' in k: del rFonts.attrib[k]
    # Footnote text: set 4pt spacing after (was 10pt)
    for style in tree.iter(f'{{{ns}}}style'):
        name_el = style.find(f'{{{ns}}}name')
        if name_el is not None and name_el.get(f'{{{ns}}}val') == 'footnote text':
            pPr = style.find(f'{{{ns}}}pPr')
            if pPr is None:
                pPr = etree.SubElement(style, f'{{{ns}}}pPr')
            spacing = pPr.find(f'{{{ns}}}spacing')
            if spacing is None:
                spacing = etree.SubElement(pPr, f'{{{ns}}}spacing')
            spacing.set(f'{{{ns}}}after', '80')  # 4pt = 80 twips
            break
    files['word/styles.xml'] = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)

# === Phase 2: Fix document.xml ===
if 'word/document.xml' in files:
    tree = etree.fromstring(files['word/document.xml'])

    for p in tree.iter(f'{{{ns}}}p'):
        pPr = p.find(f'{{{ns}}}pPr')
        if pPr is None:
            pPr = etree.Element(f'{{{ns}}}pPr')
            p.insert(0, pPr)

        # Check paragraph style
        pStyle = pPr.find(f'{{{ns}}}pStyle')
        style_val = pStyle.get(f'{{{ns}}}val') if pStyle is not None else None

        # Apply container styling for paragraphs with ''注意'' style
        if style_val == '\\u6ce8\\u610f':  # 注意
            # Add background shading
            shd = pPr.find(f'{{{ns}}}shd')
            if shd is None:
                shd = etree.SubElement(pPr, f'{{{ns}}}shd')
            shd.set(f'{{{ns}}}val', 'clear')
            shd.set(f'{{{ns}}}color', 'auto')
            shd.set(f'{{{ns}}}fill', 'f0faf5')
            # Add left border
            pBdr = pPr.find(f'{{{ns}}}pBdr')
            if pBdr is None:
                pBdr = etree.SubElement(pPr, f'{{{ns}}}pBdr')
            left = pBdr.find(f'{{{ns}}}left')
            if left is None:
                left = etree.SubElement(pBdr, f'{{{ns}}}left')
            left.set(f'{{{ns}}}val', 'single')
            left.set(f'{{{ns}}}sz', '24')
            left.set(f'{{{ns}}}space', '4')
            left.set(f'{{{ns}}}color', '42b983')
            # Add left indent
            ind = pPr.find(f'{{{ns}}}ind')
            if ind is None:
                ind = etree.SubElement(pPr, f'{{{ns}}}ind')
            ind.set(f'{{{ns}}}left', '360')
            # Ensure 1.5x line spacing (same as Normal)
            spacing = pPr.find(f'{{{ns}}}spacing')
            if spacing is None:
                spacing = etree.SubElement(pPr, f'{{{ns}}}spacing')
            spacing.set(f'{{{ns}}}line', '360')
            spacing.set(f'{{{ns}}}lineRule', 'auto')

        # Fix fonts in runs
        for rPr in p.iter(f'{{{ns}}}rPr'):
            rFonts = rPr.find(f'{{{ns}}}rFonts')
            if rFonts is not None:
                for k in list(rFonts.attrib.keys()):
                    if 'Theme' in k:
                        del rFonts.attrib[k]
                # Remove empty rFonts
                has_font = any(v for k, v in rFonts.attrib.items()
                              if any(f in k for f in ['ascii', 'hAnsi', 'eastAsia', 'cs']))
                if not has_font:
                    rPr.remove(rFonts)

        # Center figure captions and images
        texts = [t.text or '' for t in p.iter(f'{{{ns}}}t')]
        full_text = ''.join(texts).strip()
        has_img = p.find(f'.//{{{wp}}}inline') is not None or p.find(f'.//{{{wp}}}anchor') is not None
        if full_text.startswith('\\u56fe\\uff1a') or full_text.startswith('\\u56fe ') or has_img:
            jc = pPr.find(f'{{{ns}}}jc')
            if jc is None:
                jc = etree.SubElement(pPr, f'{{{ns}}}jc')
                jc.set(f'{{{ns}}}val', 'center')

    # === First-line indent for normal paragraphs ===
    # 仅正文段落缩进，排除标题/代码/容器/列表/表格
    skip_styles = {'Title', 'Subtitle', 'SourceCode', 'Source Code', '注意',
                   'Figure', 'Image Caption', 'Table Caption', 'Caption'}
    heading_ids = {'2','3','5','6','7','8','9','10','11'}  # Heading 1-9 style IDs
    title_id = '15'  # Title style
    for p in tree.iter(f'{{{ns}}}p'):
        pPr = p.find(f'{{{ns}}}pPr')
        if pPr is None: continue
        pStyle = pPr.find(f'{{{ns}}}pStyle')
        if pStyle is not None:
            val = pStyle.get(f'{{{ns}}}val')
            if val in skip_styles or val in heading_ids or val == title_id: continue
            if val and 'heading' in val.lower(): continue
        # Skip centered paragraphs (figure captions, images)
        jc = pPr.find(f'{{{ns}}}jc')
        if jc is not None and jc.get(f'{{{ns}}}val') == 'center': continue
        # Skip list items (have numPr element)
        if pPr.find(f'{{{ns}}}numPr') is not None: continue
        # Skip paragraphs inside tables
        parent = p.getparent()
        if parent is not None and parent.tag == f'{{{ns}}}tc': continue
        ind = pPr.find(f'{{{ns}}}ind')
        if ind is None:
            ind = etree.SubElement(pPr, f'{{{ns}}}ind')
        ind.set(f'{{{ns}}}firstLine', '480')

    # === Table processing: center tables + bold headers ===
    for tbl in tree.iter(f'{{{ns}}}tbl'):
        tblPr = tbl.find(f'{{{ns}}}tblPr')
        if tblPr is None:
            tblPr = etree.Element(f'{{{ns}}}tblPr')
            tbl.insert(0, tblPr)
        # Center table
        jc = tblPr.find(f'{{{ns}}}jc')
        if jc is None:
            jc = etree.SubElement(tblPr, f'{{{ns}}}jc')
        jc.set(f'{{{ns}}}val', 'center')

        # Table width 100%
        tblW = tblPr.find(f'{{{ns}}}tblW')
        if tblW is None:
            tblW = etree.SubElement(tblPr, f'{{{ns}}}tblW')
        tblW.set(f'{{{ns}}}w', '5000')
        tblW.set(f'{{{ns}}}type', 'pct')

        # Bold + center the header row
        first_row = tbl.find(f'{{{ns}}}tr')
        if first_row is not None:
            for rPr in first_row.iter(f'{{{ns}}}rPr'):
                b = rPr.find(f'{{{ns}}}b')
                if b is None:
                    b = etree.SubElement(rPr, f'{{{ns}}}b')
                b.set(f'{{{ns}}}val', 'true')
            # Center each cell vertically + horizontally
            for tc in first_row.iter(f'{{{ns}}}tc'):
                # Vertical centering
                tcPr = tc.find(f'{{{ns}}}tcPr')
                if tcPr is None:
                    tcPr = etree.Element(f'{{{ns}}}tcPr')
                    tc.insert(0, tcPr)
                vAlign = tcPr.find(f'{{{ns}}}vAlign')
                if vAlign is None:
                    vAlign = etree.SubElement(tcPr, f'{{{ns}}}vAlign')
                vAlign.set(f'{{{ns}}}val', 'center')
                # Horizontal centering for each paragraph in cell
                for p in tc.iter(f'{{{ns}}}p'):
                    pPr = p.find(f'{{{ns}}}pPr')
                    if pPr is None:
                        pPr = etree.Element(f'{{{ns}}}pPr')
                        p.insert(0, pPr)
                    jc = pPr.find(f'{{{ns}}}jc')
                    if jc is None:
                        jc = etree.SubElement(pPr, f'{{{ns}}}jc')
                    jc.set(f'{{{ns}}}val', 'center')

    # === Code block styling ===
    for p in tree.iter(f'{{{ns}}}p'):
        pPr = p.find(f'{{{ns}}}pPr')
        if pPr is None: continue
        pStyle = pPr.find(f'{{{ns}}}pStyle')
        if pStyle is None or pStyle.get(f'{{{ns}}}val') not in ('SourceCode', 'Source Code'): continue

        # Background + border
        shd = pPr.find(f'{{{ns}}}shd')
        if shd is None:
            shd = etree.SubElement(pPr, f'{{{ns}}}shd')
        shd.set(f'{{{ns}}}val', 'clear')
        shd.set(f'{{{ns}}}color', 'auto')
        shd.set(f'{{{ns}}}fill', 'f5f5f5')
        pBdr = pPr.find(f'{{{ns}}}pBdr')
        if pBdr is None:
            pBdr = etree.SubElement(pPr, f'{{{ns}}}pBdr')
        for side in ['left', 'right', 'top', 'bottom']:
            b = pBdr.find(f'{{{ns}}}{side}')
            if b is None:
                b = etree.SubElement(pBdr, f'{{{ns}}}{side}')
            b.set(f'{{{ns}}}val', 'single')
            b.set(f'{{{ns}}}sz', '4')
            b.set(f'{{{ns}}}space', '1')
            b.set(f'{{{ns}}}color', 'd0d0d0')

        # Font: Courier New (EN) + 楷体 (CN)
        for rPr in p.iter(f'{{{ns}}}rPr'):
            rFonts = rPr.find(f'{{{ns}}}rFonts')
            if rFonts is None:
                rFonts = etree.SubElement(rPr, f'{{{ns}}}rFonts')
            rFonts.set(f'{{{ns}}}ascii', 'Courier New')
            rFonts.set(f'{{{ns}}}hAnsi', 'Courier New')
            rFonts.set(f'{{{ns}}}eastAsia', '楷体')

    files['word/document.xml'] = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)

    # === URL decode footnote text ===
    if 'word/footnotes.xml' in files:
        from urllib.parse import unquote
        fn_tree = etree.fromstring(files['word/footnotes.xml'])
        for t in fn_tree.iter(f'{{{ns}}}t'):
            if t.text and '%' in t.text:
                t.text = unquote(t.text)
        files['word/footnotes.xml'] = etree.tostring(fn_tree, xml_declaration=True, encoding='UTF-8', standalone=True)

    # === Image width 100% (content width ≈ 160mm ≈ 5760000 EMU) ===
    if 'word/document.xml' in files:
        tree = etree.fromstring(files['word/document.xml'])
        wp_ns = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
        a_ns = 'http://schemas.openxmlformats.org/drawingml/2006/main'
        content_width = 5760000  # EMU ≈ 160mm

        for extent in tree.iter(f'{{{wp_ns}}}extent'):
            cx = int(extent.get('cx', 0))
            cy = int(extent.get('cy', 0))
            if cx > 0 and cy > 0:
                # Scale proportionally
                ratio = content_width / cx
                extent.set('cx', str(content_width))
                extent.set('cy', str(int(cy * ratio)))

        files['word/document.xml'] = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)

# === Write back ===
tmp = path + '.tmp'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, data in files.items():
        zout.writestr(name, data)
os.replace(tmp, path)
`);
  execFileSync('python3', [fixScript], { stdio: 'pipe', timeout: 15000 });
  unlinkSync(fixScript);
}

// 后处理：居中图片和图题（后续实现）

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function sanitizeFileName(name) {
  return name.replace(/[<>:"/\\|?*]/g, '_').substring(0, 50);
}
