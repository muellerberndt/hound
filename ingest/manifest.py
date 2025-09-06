"""Repository ingestion and card creation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Card:
    """A chunk of code with metadata."""
    id: str
    relpath: str
    char_start: int
    char_end: int
    content: str
    peek_head: str  # First 100 chars
    peek_tail: str  # Last 100 chars
    shingle_hash: str  # MinHash signature
    top_tokens: list[str]  # Most common tokens
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FileInfo:
    """Information about a source file."""
    relpath: str
    size: int
    card_ids: list[str]
    language: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class RepositoryManifest:
    """Manages repository ingestion and card creation."""
    
    def __init__(self, repo_path: str, config: dict, file_filter: list[str] | None = None, manual_chunking: bool = False):
        """Initialize manifest with repository path and config.
        
        Args:
            repo_path: Path to repository to analyze
            config: Configuration dictionary
            file_filter: Optional list of relative file paths to include (if None, includes all files)
        """
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.file_filter = file_filter  # List of specific files to include
        self.manual_chunking = manual_chunking
        self.cards: list[Card] = []
        self.files: list[FileInfo] = []
        self.file_extensions = self._get_file_extensions()
        
        # Bundling parameters
        bundle_config = config.get("bundling", {})
        self.min_chunk = bundle_config.get("min_chunk_chars", 1000)
        self.max_chunk = bundle_config.get("max_chunk_chars", 2000)
        self.target_bundle = bundle_config.get("target_chars", 25000)
    
    def _get_file_extensions(self) -> set[str]:
        """Get file extensions to process based on config."""
        # Default extensions for common languages
        defaults = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".sol", ".rs", ".go",
            ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php",
            ".swift", ".kt", ".scala", ".ml", ".hs", ".clj", ".ex", ".exs",
            ".vy", ".cairo", ".move"  # Add Vyper, Cairo, and Move support
        }
        # Add any custom extensions from config
        custom = set(self.config.get("file_extensions", []))
        return defaults | custom
    
    def walk_repository(self) -> tuple[list[Card], list[FileInfo]]:
        """Walk repository and create cards from files."""
        self.cards = []
        self.files = []
        
        for file_path in self._find_source_files():
            cards = self._process_file(file_path)
            if cards:
                self.cards.extend(cards)
                file_info = FileInfo(
                    relpath=str(file_path.relative_to(self.repo_path)),
                    size=file_path.stat().st_size,
                    card_ids=[c.id for c in cards],
                    language=self._detect_language(file_path)
                )
                self.files.append(file_info)
        
        return self.cards, self.files
    
    def _find_source_files(self) -> list[Path]:
        """Find all source files in repository."""
        files = []
        
        # If file_filter is provided, only process those specific files
        if self.file_filter:
            for file_path_str in self.file_filter:
                file_path = self.repo_path / file_path_str
                if file_path.exists() and file_path.is_file():
                    # Check extension if we have filters
                    if file_path.suffix in self.file_extensions:
                        files.append(file_path)
            return sorted(files)
        
        # Otherwise, find all source files in repository
        # Common directories to skip
        skip_dirs = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__",
            ".pytest_cache", ".venv", "venv", "env", "dist", "build",
            "target", ".idea", ".vscode", ".DS_Store"
        }
        
        for path in self.repo_path.rglob("*"):
            # Skip directories and non-files
            if path.is_dir():
                continue
            
            # Skip if in ignored directory
            if any(skip in path.parts for skip in skip_dirs):
                continue
            
            # Check extension
            if path.suffix in self.file_extensions:
                files.append(path)
        
        return sorted(files)
    
    def _process_file(self, file_path: Path) -> list[Card]:
        """Process a single file into cards."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        
        # Skip empty files
        if len(content) < 10:
            return []
        
        relpath = str(file_path.relative_to(self.repo_path))
        
        # Split into chunks at natural boundaries
        raw_chunks = self._split_into_chunks(content, file_path)
        
        cards = []
        
        if self.manual_chunking:
            chunks_with_pos = raw_chunks  # Already list of (content, start, end)
        else:
            chunks_with_pos = []
            char_offset = 0
            for chunk in raw_chunks:
                chunk_len = len(chunk)
                if not chunk.strip():
                    char_offset += chunk_len
                    continue
                chunks_with_pos.append((chunk, char_offset, char_offset + chunk_len))
                char_offset += chunk_len
        
        for i, (chunk, char_start, char_end) in enumerate(chunks_with_pos):
            card_id = self._generate_card_id(relpath, i, chunk)
            
            # Extract metadata
            peek_head = chunk[:100] if len(chunk) > 100 else chunk
            peek_tail = chunk[-100:] if len(chunk) > 100 else ""
            
            card = Card(
                id=card_id,
                relpath=relpath,
                char_start=char_start,
                char_end=char_end,
                content=chunk,
                peek_head=peek_head,
                peek_tail=peek_tail,
                shingle_hash=self._compute_shingle_hash(chunk),
                top_tokens=self._extract_top_tokens(chunk)
            )
            
            cards.append(card)
        
        return cards
    
    def _split_into_chunks(self, content: str, file_path: Path) -> list:
        """Split content into chunks at natural boundaries."""
        if self.manual_chunking:
            return self._split_from_markers(content)
        else:
            chunks = []
            current_chunk = []
            current_size = 0
            
            lines = content.split('\n')
            
            for line in lines:
                line_with_newline = line + '\n'
                line_size = len(line_with_newline)
                
                # If adding this line would exceed max chunk size, save current chunk
                if current_size + line_size > self.max_chunk and current_size > 0:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(line_with_newline)
                current_size += line_size
                
                # If we've reached a good chunk size and hit a blank line, split here
                if current_size >= self.min_chunk and not line.strip():
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            # Add remaining content
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            return chunks

    def _split_from_markers(self, content: str) -> list[tuple[str, int, int]]:
            """Split content into chunks based on manual markers '>>>CHUNK_BREAK<<<' generically, excluding markers.

            This works across any file type, detecting the marker substring in lines (after stripping whitespace),
            regardless of surrounding comment syntax (e.g., '//', '#', etc.).

            Returns:
                List of (chunk_content, char_start, char_end) tuples with non-empty chunks and original positions.
            """
            lines = content.splitlines(keepends=True)
            chunks: list[tuple[str, int, int]] = []
            current_chunk = ''
            marker = '>>>CHUNK_BREAK<<<'
            marker_count = 0
            char_pos = 0
            chunk_start = 0
            for line in lines:
                line_len = len(line)
                # Check if the line contains the marker (allowing for leading/trailing whitespace or comments)
                if marker in line.strip():
                    if current_chunk.strip():  # Only append non-empty chunks
                        chunks.append((current_chunk, chunk_start, char_pos))
                    current_chunk = ''
                    marker_count += 1
                    chunk_start = char_pos + line_len  # Start next chunk after this marker line
                else:
                    if not current_chunk:  # Starting a new chunk
                        chunk_start = char_pos
                    current_chunk += line
                char_pos += line_len
            if current_chunk.strip():  # Append the final non-empty chunk
                chunks.append((current_chunk, chunk_start, char_pos))
            if not chunks and content.strip():
                # Fallback: if no chunks were created but content exists, return it as a single chunk
                chunks = [(content, 0, len(content))]
            return chunks

    def _generate_card_id(self, relpath: str, index: int, content: str) -> str:
        """Generate unique card ID."""
        # Use sha256 for hashing
        hasher = hashlib.sha256()
        hasher.update(f"{relpath}:{index}:{content[:100]}".encode())
        hash_hex = hasher.hexdigest()[:12]
        return f"card_{hash_hex}"
    
    def _compute_shingle_hash(self, content: str) -> str:
        """Compute shingle hash for similarity comparison."""
        # Simple character n-gram approach
        n = 5  # 5-gram shingles
        shingles = set()
        
        # Clean content
        clean = ''.join(c.lower() if c.isalnum() else ' ' for c in content)
        words = clean.split()
        
        # Generate shingles from words
        text = ' '.join(words)
        for i in range(len(text) - n + 1):
            shingles.add(text[i:i+n])
        
        # Create a simple hash representation
        if shingles:
            # Sort and take first 20 shingles as signature
            sorted_shingles = sorted(shingles)[:20]
            signature = '|'.join(sorted_shingles)
            hasher = hashlib.sha256()
            hasher.update(signature.encode())
            return hasher.hexdigest()[:16]
        
        return "0" * 16
    
    def _extract_top_tokens(self, content: str, max_tokens: int = 10) -> list[str]:
        """Extract most common tokens from content."""
        # Simple tokenization
        clean = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in content)
        tokens = clean.lower().split()
        
        # Count frequencies
        freq: dict[str, int] = {}
        for token in tokens:
            if len(token) > 2:  # Skip very short tokens
                freq[token] = freq.get(token, 0) + 1
        
        # Get top tokens
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [token for token, _ in sorted_tokens[:max_tokens]]
    
    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".sol": "solidity",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".ml": "ocaml",
            ".hs": "haskell",
            ".clj": "clojure",
            ".ex": "elixir",
            ".exs": "elixir"
        }
        return ext_map.get(file_path.suffix)
    
    def save_manifest(self, output_dir: Path):
        """Save manifest to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest metadata
        manifest = {
            "repo_path": str(self.repo_path),
            "num_files": len(self.files),
            "num_cards": len(self.cards),
            "total_chars": sum(len(c.content) for c in self.cards)
        }
        # Persist whitelist used for ingestion (if any)
        if self.file_filter:
            try:
                manifest["whitelist"] = sorted(list(self.file_filter))
            except Exception:
                pass
        
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Save cards (using JSON lines for efficiency)
        with open(output_dir / "cards.jsonl", "w") as f:
            for card in self.cards:
                # Don't save full content in manifest
                card_data = card.to_dict()
                card_data.pop("content")  # Remove content to save space
                f.write(json.dumps(card_data) + "\n")
        
        # Save file info
        with open(output_dir / "files.json", "w") as f:
            json.dump([f.to_dict() for f in self.files], f, indent=2)
        
        return manifest
