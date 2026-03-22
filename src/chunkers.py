from abc import ABC, abstractmethod
from typing import List, Tuple
import ast
import re

from models import MinimalSource


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
		self.overlap = overlap

    @abstractmethod
    def chunk(self, file_path: str, content: str) -> list[tuple[MinimalSource, str]]:
        pass

	def _split_large_text(self, text: str, start_offset: int, file_path: str) -> List[tuple[MinimalSource, str]]:
		sub_chunks = []
		current_pos = 0
		
		while current_pos < len(text):
			end_pos = min(current_pos + self.max_chunk_size, len(text))

			sub_text = text[current_pos:end_pos]

			source = MinimalSource(
				file_path=file_path,
				first_character_index=start_offset + current_pos,
				last_character_index=start_offset + end_pos
			)
			
			sub_chunks.append((source, sub_text))

			if end_pos == len(text):
				break
			current_pos += (self.max_chunk_size - self.overlap)

		return sub_chunks


class PythonChunker(BaseChunker):
    def _build_line_map(self, content: str) -> list[int]:
        line_starts = [0]
        for i, char in enumerate(content):
            if char == '\n':
                line_starts.append(i + 1)
        return line_starts

    def _line_to_char(self, line_map: list[int], line: int | None, col: int | None) -> int:
        if line is None or col is None:
            return 0
        return line_map[line - 1] + col

    def chunk(self, file_path: str, content: str) -> List[Tuple[MinimalSource, str]]:
        chunks: List[Tuple[MinimalSource, str]] = []
        line_map = self._build_line_map(content)
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._split_large_text(content, 0, file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
				end_line = node.end_lineno if node.end_lineno is not None else node.lineno
                end_col = node.end_col_offset if node.end_col_offset is not None else 0

                start_char = self._line_to_char(line_map, node.lineno, node.col_offset)
                end_char = self._line_to_char(line_map, end_line, end_col)

                chunk_text = content[start_char:end_char]
                
                if len(chunk_text) > self.max_chunk_size:
					sub_chunks = self._split_large_text(chunk_text, start_char, file_path)
                    chunks.extend(sub_chunks)
                else:
					source = MinimalSource(
						file_path=file_path,
						first_character_index=start_char,
						last_character_index=end_char
					)
					chunks.append((source, chunk_text))
        
		if not chunks and content.strip():
            return self._split_large_text(content, 0, file_path)

        return chunks

class MdChunker(BaseChunker):
    def chunk(self, file_path: str, content: str) -> List[Tuple[MinimalSource, str]]:
        chunks: List[Tuple[MinimalSource, str]] = []
        headers = list(re.finditer(r'^(#{1,6}\s+.+)$', content, re.MULTILINE))
        
        if not headers:
            return self._split_large_text(content, 0, file_path)

		if headers[0].start() > 0:
            intro_text = content[0:headers[0].start()]
            if intro_text.strip():
                chunks.extend(self._split_large_text(intro_text, 0, file_path))

        for i in range(len(headers)):
            start_pos = headers[i].start()

            if i + 1 < len(headers):
                end_pos = headers[i+1].start()
            else:
                end_pos = len(content)

            section_text = content[start_pos:end_pos]

            if len(section_text) > self.max_chunk_size:
                chunks.extend(self._split_large_text(section_text, start_pos, file_path))
            else:
                source = MinimalSource(
                    file_path=file_path,
                    first_character_index=start_pos,
                    last_character_index=end_pos
                )
                chunks.append((source, section_text))
                
        return chunks