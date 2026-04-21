#!/usr/bin/env node
/**
 * 从文档中提取标记的类定义，同步到共享模块目录
 *
 * 用法: node scripts/extract-shared-modules.js
 *
 * 标记语法: ```python runnable extract-class="ClassName"
 *
 * 路径映射（支持多级目录）:
 *   docs/statistical-learning/linear-models/*.md → shared_modules/linear/*.py
 *   docs/statistical-learning/bayesian-methods/*.md → shared_modules/bayesian/*.py
 *   docs/statistical-learning/support-vector-machines/*.md → shared_modules/svm/*.py
 *   docs/statistical-learning/decision-tree-ensemble/*.md → shared_modules/tree/*.py
 *   docs/statistical-learning/unsupervised-learning/*.md → shared_modules/unsupervised/*.py
 *   docs/deep-learning/neural-network-structure/*.md → shared_modules/neural/*.py
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 项目根目录
const PROJECT_ROOT = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(PROJECT_ROOT, 'docs');
const SHARED_MODULES_DIR = path.join(PROJECT_ROOT, 'local-server', 'shared_modules');

// 文档路径到模块路径的映射（支持多级目录）
const CHAPTER_MAPPING = {
  // statistical-learning
  'statistical-learning/linear-models': 'linear',
  'statistical-learning/bayesian-methods': 'bayesian',
  'statistical-learning/support-vector-machines': 'svm',
  'statistical-learning/decision-tree-ensemble': 'tree',
  'statistical-learning/unsupervised-learning': 'unsupervised',
  // deep-learning
  'deep-learning/neural-network-structure': 'neural'
};

// 类名到文件名的转换 (PascalCase → snake_case)
function classNameToFileName(className) {
  return className
    .replace(/([A-Z])/g, '_$1')
    .toLowerCase()
    .replace(/^_/, '');
}

// 改进的类定义提取（处理嵌套结构）
function extractClassDefinition(code, className) {
  const lines = code.split('\n');
  const result = [];
  let foundClass = false;
  let classBaseIndent = -1;
  let inMethod = false;
  let methodBaseIndent = -1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    const lineIndent = line.search(/\S/);

    // 跳过开头的 import 语句（不属于类定义）
    if (!foundClass && (/^(import|from)\s+/.test(trimmedLine) || trimmedLine === '')) {
      continue;
    }

    // 查找目标类的定义
    if (!foundClass) {
      const classMatch = trimmedLine.match(/^class\s+(\w+)/);
      if (classMatch && classMatch[1] === className) {
        foundClass = true;
        classBaseIndent = lineIndent;
        result.push(line);
        continue;
      }
    } else {
      // 已经找到类定义，收集类内容

      // 空行直接添加
      if (trimmedLine === '') {
        result.push(line);
        continue;
      }

      // 检查是否是类方法定义
      if (lineIndent === classBaseIndent + 4 && /^def\s+/.test(trimmedLine)) {
        inMethod = true;
        methodBaseIndent = lineIndent;
        result.push(line);
        continue;
      }

      // 检查是否遇到同级的顶级定义（类结束）
      if (lineIndent === classBaseIndent) {
        // 遇到新的顶级 class 或顶层代码，类定义结束
        if (/^class\s+\w/.test(trimmedLine) || /^def\s+\w/.test(trimmedLine)) {
          break;
        }
        // 顶层的注释或变量赋值也结束类定义（如 # 测试代码）
        break;
      }

      // 检查是否是顶层代码（比类缩进更少）
      if (lineIndent < classBaseIndent) {
        break;
      }

      result.push(line);
    }
  }

  // 移除末尾多余的空行
  while (result.length > 0 && result[result.length - 1].trim() === '') {
    result.pop();
  }

  return result.join('\n');
}

// 处理单个文件
function processFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const results = [];

  // 正则匹配 extract-class="ClassName"
  const codeBlockRegex = /```python\s+runnable\s+extract-class="(\w+)"\s*\n([\s\S]*?)```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    const className = match[1];
    const code = match[2];
    const classDefinition = extractClassDefinition(code, className);

    if (classDefinition) {
      results.push({
        className,
        code: classDefinition
      });
      console.log(`  ✓ 找到类: ${className}`);
    } else {
      console.log(`  ⚠ 未能提取类: ${className}`);
    }
  }

  return results;
}

// 生成 Python 模块文件
function generateModuleFile(className, classCode, moduleDir) {
  const fileName = classNameToFileName(className) + '.py';
  const modulePath = path.join(SHARED_MODULES_DIR, moduleDir, fileName);

  // 检测是否需要 import numpy
  const needsNumpy = classCode.includes('np.') || classCode.includes('numpy');

  let imports = '';
  if (needsNumpy) {
    imports = 'import numpy as np\n\n';
  }

  const content = `# ${className} 类定义
# 从文档自动提取生成

${imports}${classCode}
`;

  return { path: modulePath, content };
}

// 更新 __init__.py
function updateInitPy(moduleDir, classNames) {
  const initPath = path.join(SHARED_MODULES_DIR, moduleDir, '__init__.py');

  if (!fs.existsSync(initPath)) {
    return;
  }

  let content = fs.readFileSync(initPath, 'utf-8');

  // 添加导入语句
  for (const className of classNames) {
    const importLine = `from .${classNameToFileName(className)} import ${className}`;
    if (!content.includes(importLine)) {
      // 在 __all__ = [] 之前添加 import
      content = content.replace(
        /(__all__\s*=\s*\[)/,
        `${importLine}\n$1`
      );
      // 添加到 __all__
      content = content.replace(
        /(__all__\s*=\s*\[)([^\]]*)(\])/,
        `$1$2, '${className}'$3`
      );
      // 清理多余的逗号
      content = content.replace(/,(\s*])/, '$1');  // 末尾逗号
      content = content.replace(/,\s*,/g, ',');    // 连续逗号
      content = content.replace(/\[,/, '[');       // 开头逗号
    }
  }

  fs.writeFileSync(initPath, content);
}

// 主函数
function main() {
  console.log('开始提取共享模块...\n');

  // 确保共享模块目录存在
  if (!fs.existsSync(SHARED_MODULES_DIR)) {
    console.log('错误: shared_modules 目录不存在');
    process.exit(1);
  }

  // 扫描所有文档
  const moduleClasses = {}; // 按模块分组收集类名

  for (const docPath of Object.keys(CHAPTER_MAPPING)) {
    const chapterPath = path.join(DOCS_DIR, docPath);

    if (!fs.existsSync(chapterPath)) {
      continue;
    }

    console.log(`\n扫描 ${docPath}/`);

    const files = fs.readdirSync(chapterPath).filter(f => f.endsWith('.md'));

    for (const file of files) {
      const filePath = path.join(chapterPath, file);
      const classes = processFile(filePath);

      if (classes.length > 0) {
        const moduleDir = CHAPTER_MAPPING[docPath];

        // 确保模块目录存在
        const fullModuleDir = path.join(SHARED_MODULES_DIR, moduleDir);
        if (!fs.existsSync(fullModuleDir)) {
          fs.mkdirSync(fullModuleDir, { recursive: true });
          // 创建 __init__.py
          fs.writeFileSync(path.join(fullModuleDir, '__init__.py'), '__all__ = []\n');
          console.log(`    创建目录: ${moduleDir}/`);
        }

        if (!moduleClasses[moduleDir]) {
          moduleClasses[moduleDir] = [];
        }

        for (const { className, code } of classes) {
          // 生成模块文件
          const { path: modulePath, content } = generateModuleFile(className, code, moduleDir);
          fs.writeFileSync(modulePath, content);
          console.log(`    写入: ${path.relative(PROJECT_ROOT, modulePath)}`);

          moduleClasses[moduleDir].push(className);
        }
      }
    }
  }

  // 更新各模块的 __init__.py
  console.log('\n更新 __init__.py 文件...');
  for (const [moduleDir, classNames] of Object.entries(moduleClasses)) {
    updateInitPy(moduleDir, [...new Set(classNames)]);
    console.log(`  ✓ ${moduleDir}/__init__.py`);
  }

  // 更新顶层 __init__.py
  const topInitPath = path.join(SHARED_MODULES_DIR, '__init__.py');
  if (fs.existsSync(topInitPath)) {
    let content = fs.readFileSync(topInitPath, 'utf-8');
    const allImports = [];

    for (const [moduleDir, classNames] of Object.entries(moduleClasses)) {
      for (const className of classNames) {
        const importLine = `from .${moduleDir}.${classNameToFileName(className)} import ${className}`;
        if (!content.includes(importLine)) {
          allImports.push(importLine);
        }
      }
    }

    if (allImports.length > 0) {
      content = allImports.join('\n') + '\n' + content;
      fs.writeFileSync(topInitPath, content);
    }
  }

  console.log('\n✓ 提取完成！');
}

main();