#!/usr/bin/env node
/**
 * 从文档中提取标记的类定义，同步到共享模块目录
 *
 * 用法: node scripts/extract-shared-modules.js
 *
 * 标记语法: ```python runnable extract-class="ClassName"
 *
 * 模块路径推断规则:
 *   1. 显式映射（CHAPTER_MAPPING）优先，用于特殊命名
 *   2. 未映射的目录自动推断：docs/<category>/<chapter>/ → shared_modules/<chapter_snake_case>/
 *
 * 显式映射示例:
 *   'neural-network-structure' → 'neural' (简化命名)
 *   'convolutional-neural-network' → 'cnn' (简化命名)
 *
 * 自动推断示例:
 *   docs/deep-learning/transformer-models/ → shared_modules/transformer_models/
 *   docs/statistical-learning/new-topic/ → shared_modules/new_topic/
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

// 显式映射（用于特殊命名覆盖）
// 只在这里添加需要简化或特殊命名的目录
const CHAPTER_MAPPING = {
  // statistical-learning - 简化命名
  'statistical-learning/linear-models': 'linear',
  'statistical-learning/bayesian-methods': 'bayesian',
  'statistical-learning/support-vector-machines': 'svm',
  'statistical-learning/decision-tree-ensemble': 'tree',
  'statistical-learning/unsupervised-learning': 'unsupervised',
  // deep-learning - 简化命名
  'deep-learning/neural-network-structure': 'neural',
  'deep-learning/convolutional-neural-network': 'cnn'
};

// 将目录名转换为 snake_case 模块名
function dirToModuleName(dirName) {
  return dirName
    .replace(/-/g, '_')
    .toLowerCase();
}

// 推断模块路径（先查显式映射，再用自动推断）
function inferModulePath(docPath) {
  // 1. 查显式映射
  if (CHAPTER_MAPPING[docPath]) {
    return CHAPTER_MAPPING[docPath];
  }

  // 2. 自动推断：取最后一级目录名转为 snake_case
  // 例如: deep-learning/transformer-models → transformer_models
  const parts = docPath.split('/');
  const lastDir = parts[parts.length - 1];
  return dirToModuleName(lastDir);
}

// 常见技术术语（保持不拆分，作为完整单词）
const KNOWN_TERMS = [
  'ImageNet', 'AlexNet', 'VGGNet', 'ResNet', 'GoogleNet', 'Inception',
  'YOLO', 'Transformer', 'Encoder', 'Decoder', 'Attention',
  'Dataset', 'DataLoader', 'Optimizer', 'Scheduler',
  'Classifier', 'Regressor', 'Network', 'Model', 'Layer',
  // 机器学习缩写词（保持完整）
  'SVM', 'PCA', 'KMeans', 'GMM', 'LDA', 'NLP', 'CNN', 'RNN', 'LSTM', 'GAN'
];

// 类名到文件名的转换 (PascalCase → snake_case)
// 保留常见术语不拆分，例如 ImageNet → imagenet（而非 image_net）
function classNameToFileName(className) {
  // 术语到占位符的映射（使用不会被拆分的格式）
  const termMap = {};
  for (let i = 0; i < KNOWN_TERMS.length; i++) {
    termMap[KNOWN_TERMS[i]] = `__TERM${i}__`;
  }

  let result = className;

  // 先用占位符保护已知术语（大写匹配，保护整个单词）
  for (const [term, placeholder] of Object.entries(termMap)) {
    result = result.replace(term, placeholder);
  }

  // 在占位符前后添加下划线边界（确保术语与其他部分分隔）
  result = result
    .replace(/([a-z])(__TERM)/g, '$1_$2')  // 小写后接占位符时加下划线
    .replace(/(__\w+__)(__TERM)/g, '$1_$2')  // 占位符后接占位符时加下划线
    .replace(/(__TERM\d+__)([A-Z])/g, '$1_$2');  // 占位符后接大写时加下划线

  // 对剩余部分进行 snake_case 转换
  result = result
    .replace(/([a-z])([A-Z])/g, '$1_$2')  // 小写后接大写时加下划线
    .toLowerCase();

  // 恢复已知术语（占位符 → 小写术语）
  for (let i = 0; i < KNOWN_TERMS.length; i++) {
    const term = KNOWN_TERMS[i].toLowerCase();
    const placeholder = `__term${i}__`;  // 占位符也被 toLowerCase 了
    result = result.replace(placeholder, term);
  }

  // 清理多余的下划线
  result = result.replace(/_+/g, '_').replace(/^_|_$/g, '');

  return result;
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
  // 支持格式: ```python runnable [gpu] [timeout=xxx] extract-class="ClassName"
  const codeBlockRegex = /```python\s+runnable[\s\w=]*extract-class="(\w+)"[\s\w=]*\n([\s\S]*?)```/g;
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

  // 自动检测需要的导入
  const imports = [];

  // PyTorch 相关
  if (classCode.includes('torch.') || classCode.includes('nn.') || classCode.includes('Tensor')) {
    imports.push('import torch');
  }
  if (classCode.includes('nn.') || classCode.includes('nn.Module') || classCode.includes('Sequential') ||
      classCode.includes('Conv2d') || classCode.includes('Linear') || classCode.includes('ReLU') ||
      classCode.includes('Dropout') || classCode.includes('MaxPool2d') || classCode.includes('AdaptiveAvgPool2d')) {
    imports.push('import torch.nn as nn');
  }
  if (classCode.includes('Dataset') || classCode.includes('DataLoader')) {
    imports.push('from torch.utils.data import Dataset, DataLoader');
  }

  // 图像处理
  if (classCode.includes('Image')) {
    imports.push('from PIL import Image');
  }
  if (classCode.includes('transforms.')) {
    imports.push('from torchvision import transforms');
  }

  // 数值计算
  if (classCode.includes('np.') || classCode.includes('numpy')) {
    imports.push('import numpy as np');
  }

  // 文件系统
  if (classCode.includes('os.') || classCode.includes('os.path')) {
    imports.push('import os');
  }

  // 排序并去重
  const uniqueImports = [...new Set(imports)].sort((a, b) => {
    // from 导入排在后面
    if (a.startsWith('from') && !b.startsWith('from')) return 1;
    if (!a.startsWith('from') && b.startsWith('from')) return -1;
    return a.localeCompare(b);
  });

  const importsSection = uniqueImports.length > 0
    ? uniqueImports.join('\n') + '\n\n'
    : '';

  const content = `# ${className} 类定义
# 从文档自动提取生成

${importsSection}${classCode}
`;

  return { path: modulePath, content };
}

// 更新 __init__.py（完全重新生成，扫描目录中所有 .py 文件）
function updateInitPy(moduleDir, classNames) {
  const initPath = path.join(SHARED_MODULES_DIR, moduleDir, '__init__.py');
  const moduleFullPath = path.join(SHARED_MODULES_DIR, moduleDir);

  if (!fs.existsSync(initPath)) {
    return;
  }

  // 扫描目录下所有 .py 文件（排除 __init__.py 和 __pycache__）
  const existingFiles = fs.readdirSync(moduleFullPath)
    .filter(f => f.endsWith('.py') && f !== '__init__.py');

  // 从 .py 文件内容中提取类名，并映射到实际文件名
  const fileClassMap = {};  // 文件名 -> 类名列表
  const allClassNames = [];

  for (const file of existingFiles) {
    const filePath = path.join(moduleFullPath, file);
    const content = fs.readFileSync(filePath, 'utf-8');
    const classMatches = content.match(/^class\s+(\w+)/gm);
    const classes = classMatches ? classMatches.map(m => m.replace('class ', '')) : [];

    if (classes.length > 0) {
      fileClassMap[file] = classes;
      allClassNames.push(...classes);
    }
  }

  // 合并新提取的类名
  allClassNames.push(...classNames);
  const uniqueClassNames = [...new Set(allClassNames)];

  // 生成导入语句
  // 对于已有文件中的类，使用实际文件名；对于新类，使用推断的文件名
  const imports = [];
  for (const className of uniqueClassNames) {
    // 查找类所在的实际文件
    let actualFileName = null;
    for (const [file, classes] of Object.entries(fileClassMap)) {
      if (classes.includes(className)) {
        actualFileName = file.replace('.py', '');
        break;
      }
    }

    // 如果没找到，使用推断的文件名
    if (!actualFileName) {
      actualFileName = classNameToFileName(className);
    }

    imports.push(`from .${actualFileName} import ${className}`);
  }

  // 生成 __all__ 列表
  const allList = uniqueClassNames.map(n => `'${n}'`).join(', ');

  const content = `# ${moduleDir.toUpperCase()} 模块
${imports.join('\n')}

__all__ = [${allList}]
`;

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

  // 扫描所有文档目录（支持自动推断）
  const moduleClasses = {}; // 按模块分组收集类名

  // 遍历 docs 下的所有子目录
  const categories = fs.readdirSync(DOCS_DIR).filter(f => {
    const fullPath = path.join(DOCS_DIR, f);
    return fs.statSync(fullPath).isDirectory() &&
           !f.startsWith('.') &&
           f !== 'superpowers';  // 排除 superpowers（内部文档）
  });

  for (const category of categories) {
    const categoryPath = path.join(DOCS_DIR, category);
    const chapters = fs.readdirSync(categoryPath).filter(f => {
      const fullPath = path.join(categoryPath, f);
      return fs.statSync(fullPath).isDirectory() && !f.startsWith('.');
    });

    for (const chapter of chapters) {
      const docPath = `${category}/${chapter}`;
      const chapterPath = path.join(DOCS_DIR, docPath);

      console.log(`\n扫描 ${docPath}/`);

      const files = fs.readdirSync(chapterPath).filter(f => f.endsWith('.md'));

      for (const file of files) {
        const filePath = path.join(chapterPath, file);
        const classes = processFile(filePath);

        if (classes.length > 0) {
          // 使用推断函数获取模块路径（支持显式映射 + 自动推断）
          const moduleDir = inferModulePath(docPath);
          const mappingType = CHAPTER_MAPPING[docPath] ? '(显式映射)' : '(自动推断)';
          console.log(`  模块路径: shared_modules/${moduleDir}/ ${mappingType}`);

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
  }

  // 更新各模块的 __init__.py
  console.log('\n更新 __init__.py 文件...');
  for (const [moduleDir, classNames] of Object.entries(moduleClasses)) {
    updateInitPy(moduleDir, [...new Set(classNames)]);
    console.log(`  ✓ ${moduleDir}/__init__.py`);
  }

  // 更新顶层 __init__.py（完全重新生成）
  const topInitPath = path.join(SHARED_MODULES_DIR, '__init__.py');
  if (fs.existsSync(topInitPath)) {
    // 获取所有子模块目录（排除 __pycache__）
    const subModules = fs.readdirSync(SHARED_MODULES_DIR).filter(f => {
      const fullPath = path.join(SHARED_MODULES_DIR, f);
      return fs.statSync(fullPath).isDirectory() &&
             !f.startsWith('.') &&
             f !== '__pycache__';
    });

    // 生成简洁的 __init__.py：只使用 from .xxx import * 模式
    const imports = subModules.map(m => `from .${m} import *`).join('\n');
    const content = `# shared 模块包初始化
# 包含统计学习系列文档中可复用的类定义

${imports}
`;

    fs.writeFileSync(topInitPath, content);
    console.log('  ✓ shared_modules/__init__.py');
  }

  console.log('\n✓ 提取完成！');
}

main();