# MarkdownChunker 定义
# 从文档自动提取生成

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field

class MarkdownChunker:
    """Markdown 文档解析与切分器

    以 Markdown 标题（# 开头行）为边界将文档切分为多个块，
    每个块包含标题层级下全部文本。保留来源、章节等元数据。
    """

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 3000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def parse_file(self, filepath: str) -> list[Chunk]:
        """解析单个 Markdown 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        source = os.path.basename(filepath)
        return self._parse_content(content, source)

    def parse_directory(self, dirpath: str,
                        exclude_files: set = None) -> list[Chunk]:
        """解析目录下所有 Markdown 文件"""
        if exclude_files is None:
            exclude_files = set()

        all_chunks = []
        file_count = 0
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.md') or fname in exclude_files:
                continue
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue
            chunks = self.parse_file(fpath)
            all_chunks.extend(chunks)
            file_count += 1

        # 递归处理子目录
        for sub in sorted(os.listdir(dirpath)):
            subpath = os.path.join(dirpath, sub)
            if os.path.isdir(subpath) and not sub.startswith('.') and sub != 'assets':
                sub_chunks = self.parse_directory(subpath, exclude_files)
                all_chunks.extend(sub_chunks)

        return all_chunks

    def _parse_content(self, content: str, source: str) -> list[Chunk]:
        """按标题边界切分文档内容"""
        chunks = []
        lines = content.split('\n')
        current_section = ''     # 当前章节标题
        current_lines = []       # 当前块内的文本行
        block_index = 0          # 文档内的绝对块序号
        seen_h1 = False          # 是否已经遇到文档一级标题
        in_code_block = False    # 是否在代码块内

        for line in lines:
            # 跟踪代码块边界（代码块内的 # 不是标题）
            if line.strip()[:3] == chr(96) * 3:
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            heading = re.match(r'^(#{1,4})\s+(.+)', line)

            if heading and not in_code_block and seen_h1:
                # 遇到新标题：保存当前块
                text = '\n'.join(current_lines).strip()
                if len(text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"{source}#{block_index}",
                        text=text,
                        source=source,
                        section=current_section or '引言',
                        chunk_index=block_index,
                    ))
                    block_index += 1

                current_section = heading.group(2)
                current_lines = []
            elif heading and not in_code_block and not seen_h1:
                # 文档一级标题：记录章节但不保存前面内容
                seen_h1 = True
                current_section = heading.group(2)
            else:
                current_lines.append(line)

        # 保存最后一个块
        if current_lines:
            text = '\n'.join(current_lines).strip()
            if len(text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{source}#{block_index}",
                    text=text,
                    source=source,
                    section=current_section or '引言',
                    chunk_index=block_index,
                ))

        return chunks
