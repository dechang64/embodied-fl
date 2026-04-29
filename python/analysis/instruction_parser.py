# ── python/analysis/instruction_parser.py ──
"""
Instruction Parser for VLA Models
====================================
Parses natural language robot instructions into structured representations
for federated task matching and training.

Capabilities:
  - Extract task type from instruction (grasping, navigation, assembly, etc.)
  - Extract object references (what to manipulate)
  - Extract spatial relations (on, in, under, next to, etc.)
  - Generate instruction embeddings for HNSW task matching
  - Multi-language support (Chinese + English)

Federated Learning Integration:
  - Instruction embeddings used for task matching in HNSW index
  - Structured parse used for task-type-aware aggregation weighting
  - Shared parser across all clients ensures consistent task understanding
"""

import re
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter


# ── Task Taxonomy ──

TASK_KEYWORDS = {
    "grasping": {
        "en": ["pick", "grab", "grasp", "lift", "take", "hold", "catch", "fetch"],
        "zh": ["抓", "拿", "拾", "握", "提取", "抓取", "拿起", "捡起"],
    },
    "placing": {
        "en": ["place", "put", "set", "lay", "drop", "release", "put down", "set down"],
        "zh": ["放", "放置", "放下", "搁", "摆", "安放"],
    },
    "navigation": {
        "en": ["go", "move", "navigate", "walk", "drive", "travel", "reach", "approach"],
        "zh": ["走", "移动", "导航", "前往", "到达", "接近", "去"],
    },
    "assembly": {
        "en": ["assemble", "insert", "attach", "connect", "screw", "mount", "install", "fit"],
        "zh": ["装配", "组装", "插入", "连接", "拧", "安装", "对接"],
    },
    "inspection": {
        "en": ["inspect", "check", "examine", "scan", "detect", "verify", "test", "measure"],
        "zh": ["检查", "检测", "检验", "扫描", "测量", "查看", "观察"],
    },
    "pouring": {
        "en": ["pour", "fill", "empty", "transfer", "spill"],
        "zh": ["倒", "灌", "注入", "倾倒"],
    },
    "cutting": {
        "en": ["cut", "slice", "chop", "carve", "trim", "shear"],
        "zh": ["切", "割", "剪", "裁", "削", "劈"],
    },
    "wiping": {
        "en": ["wipe", "clean", "sweep", "wash", "scrub", "dust", "mop"],
        "zh": ["擦", "清洁", "扫", "洗", "刷", "抹"],
    },
    "stacking": {
        "en": ["stack", "pile", "arrange", "organize", "sort", "group"],
        "zh": ["堆", "叠", "排列", "整理", "分类", "码放"],
    },
    "opening": {
        "en": ["open", "unlock", "unlatch", "uncap", "uncover"],
        "zh": ["打开", "开", "解锁", "掀开", "揭开"],
    },
    "closing": {
        "en": ["close", "shut", "lock", "latch", "cap", "cover", "seal"],
        "zh": ["关闭", "关", "锁", "盖上", "封"],
    },
    "pushing": {
        "en": ["push", "press", "shove", "nudge"],
        "zh": ["推", "按", "压", "挤"],
    },
    "pulling": {
        "en": ["pull", "drag", "tug", "yank", "draw"],
        "zh": ["拉", "拖", "拽", "扯", "抽"],
    },
    "custom": {
        "en": [],
        "zh": [],
    },
}

# Spatial relation keywords
SPATIAL_KEYWORDS = {
    "on": {"en": ["on", "on top of", "onto"], "zh": ["上", "上面", "上方"]},
    "in": {"en": ["in", "inside", "into", "within"], "zh": ["里", "里面", "内部", "中"]},
    "under": {"en": ["under", "below", "beneath", "underneath"], "zh": ["下", "下面", "下方"]},
    "next_to": {"en": ["next to", "beside", "alongside", "by"], "zh": ["旁边", "边上", "侧"]},
    "behind": {"en": ["behind", "in back of"], "zh": ["后面", "背后"]},
    "in_front": {"en": ["in front of", "before", "facing"], "zh": ["前面", "前方"]},
    "between": {"en": ["between", "among"], "zh": ["之间", "中间"]},
}


@dataclass
class ParsedInstruction:
    """Structured parse result of a robot instruction."""
    raw: str = ""
    language: str = "en"             # "en", "zh", or "mixed"
    task_type: str = "custom"        # Primary task type
    task_confidence: float = 0.0     # Confidence in task classification
    objects: list[str] = field(default_factory=list)    # Referenced objects
    spatial_relations: list[dict] = field(default_factory=list)  # [{relation, object}]
    has_negation: bool = False       # Contains "don't", "not", "不要", etc.
    embedding_hash: str = ""         # Deterministic hash for HNSW indexing

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "language": self.language,
            "task_type": self.task_type,
            "task_confidence": self.task_confidence,
            "objects": self.objects,
            "spatial_relations": self.spatial_relations,
            "has_negation": self.has_negation,
            "embedding_hash": self.embedding_hash,
        }


class InstructionParser:
    """Parse natural language robot instructions.

    Usage:
        parser = InstructionParser()
        result = parser.parse("pick up the red cup and put it on the table")
        # result.task_type = "grasping"
        # result.objects = ["red cup", "table"]
        # result.spatial_relations = [{"relation": "on", "object": "table"}]
    """

    def __init__(self, custom_tasks: Optional[dict] = None):
        self.task_keywords = dict(TASK_KEYWORDS)
        if custom_tasks:
            for task, keywords in custom_tasks.items():
                if task not in self.task_keywords:
                    self.task_keywords[task] = keywords

        # Compile regex patterns for negation
        self._negation_pattern = re.compile(
            r"\b(don'?t|do not|not|never|no|不要|别|不能|禁止)\b",
            re.IGNORECASE,
        )

        # Spatial relation pattern
        self._spatial_pattern = re.compile(
            r"(\w+)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )

    def parse(self, instruction: str) -> ParsedInstruction:
        """Parse a natural language instruction.

        Args:
            instruction: Raw instruction string.

        Returns:
            ParsedInstruction with structured information.
        """
        if not instruction or not instruction.strip():
            return ParsedInstruction(raw=instruction)

        # Detect language
        language = self._detect_language(instruction)

        # Detect task type
        task_type, confidence = self._classify_task(instruction, language)

        # Extract objects
        objects = self._extract_objects(instruction, language)

        # Extract spatial relations
        spatial = self._extract_spatial_relations(instruction, language)

        # Detect negation
        has_negation = bool(self._negation_pattern.search(instruction))

        # Compute embedding hash
        embedding_hash = self._compute_hash(instruction)

        return ParsedInstruction(
            raw=instruction,
            language=language,
            task_type=task_type,
            task_confidence=confidence,
            objects=objects,
            spatial_relations=spatial,
            has_negation=has_negation,
            embedding_hash=embedding_hash,
        )

    def parse_batch(self, instructions: list[str]) -> list[ParsedInstruction]:
        """Parse multiple instructions.

        Args:
            instructions: List of instruction strings.

        Returns:
            List of ParsedInstruction objects.
        """
        return [self.parse(inst) for inst in instructions]

    def _detect_language(self, text: str) -> str:
        """Detect if text is Chinese, English, or mixed."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())

        if chinese_chars == 0:
            return "en"
        elif latin_chars == 0:
            return "zh"
        else:
            return "mixed"

    def _classify_task(
        self, instruction: str, language: str
    ) -> tuple[str, float]:
        """Classify the primary task type from instruction.

        Returns:
            (task_type, confidence) tuple.
        """
        instruction_lower = instruction.lower()
        scores: dict[str, float] = {}

        for task_type, keywords_by_lang in self.task_keywords.items():
            score = 0.0
            matched = 0

            # Check keywords for detected language(s)
            langs_to_check = [language]
            if language == "mixed":
                langs_to_check = ["en", "zh"]

            for lang in langs_to_check:
                keywords = keywords_by_lang.get(lang, [])
                for kw in keywords:
                    if kw in instruction_lower:
                        score += 1.0
                        matched += 1

            if matched > 0:
                # Normalize by number of keywords for this task
                total_kw = sum(
                    len(keywords_by_lang.get(l, [])) for l in langs_to_check
                )
                scores[task_type] = score / max(total_kw, 1)

        if not scores:
            return "custom", 0.0

        best_task = max(scores, key=scores.get)
        best_score = scores[best_task]

        # Confidence: how much better is the best vs second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(best_score * (1 + gap), 1.0)
        else:
            confidence = min(best_score, 1.0)

        return best_task, round(confidence, 3)

    def _extract_objects(
        self, instruction: str, language: str
    ) -> list[str]:
        """Extract object references from instruction.

        This is a simple heuristic extraction. For production,
        use NER (Named Entity Recognition) or LLM-based extraction.
        """
        objects = []

        # English: look for noun phrases after articles
        # Split on conjunctions first, then extract per clause
        clauses = re.split(r"\b(?:and|or|but)\b", instruction)
        stop_words = {
            "on", "in", "under", "to", "from", "with", "into", "onto",
            "beside", "behind", "before", "after", "above", "below",
            "between", "through", "during", "until", "against", "about", "than",
        }
        for clause in clauses:
            en_pattern = re.compile(
                r"(?:the|a|an)\s+(\w+(?:\s+\w+)*)",
                re.IGNORECASE,
            )
            for m in en_pattern.finditer(clause):
                phrase = m.group(1).strip()
                # Truncate at stop words
                words = phrase.split()
                truncated = []
                for w in words:
                    if w.lower() in stop_words:
                        break
                    truncated.append(w)
                if truncated:
                    final = " ".join(truncated)
                    skip_words = {"it", "them", "this", "that", "there", "here"}
                    if truncated[-1].lower() not in skip_words:
                        objects.append(final)

        # Chinese: look for characters after action verbs
        if language in ("zh", "mixed"):
            # Pattern 1: noun before particles
            zh_objects = re.findall(r"[\u4e00-\u9fff]{1,4}(?=的|和|到|在|上|下|里)", instruction)
            objects.extend(zh_objects)
            # Pattern 2: noun phrase at end of instruction (after verb)
            zh_tail = re.findall(r"[\u4e00-\u9fff]{2,6}$", instruction)
            objects.extend(zh_tail)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for obj in objects:
            obj_lower = obj.lower()
            if obj_lower not in seen:
                seen.add(obj_lower)
                unique.append(obj)

        return unique

    def _extract_spatial_relations(
        self, instruction: str, language: str
    ) -> list[dict]:
        """Extract spatial relations from instruction."""
        relations = []
        instruction_lower = instruction.lower()

        for relation, keywords_by_lang in SPATIAL_KEYWORDS.items():
            langs_to_check = [language]
            if language == "mixed":
                langs_to_check = ["en", "zh"]

            for lang in langs_to_check:
                keywords = keywords_by_lang.get(lang, [])
                for kw in keywords:
                    if kw in instruction_lower:
                        # Try to find the object after the spatial keyword
                        idx = instruction_lower.find(kw)
                        after = instruction_lower[idx + len(kw):].strip()
                        # Extract first noun phrase
                        obj_match = re.match(
                            r"(?:the\s+|a\s+|an\s+)?(\w+)",
                            after,
                        )
                        obj = obj_match.group(1) if obj_match else ""
                        if obj:
                            relations.append({
                                "relation": relation,
                                "object": obj,
                            })
                        break  # One match per relation type

        return relations

    def _compute_hash(self, text: str) -> str:
        """Compute deterministic hash for instruction embedding."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get_task_distribution(
        self, instructions: list[str]
    ) -> dict[str, int]:
        """Count task type distribution across instructions.

        Args:
            instructions: List of instruction strings.

        Returns:
            Dict mapping task_type to count.
        """
        parsed = self.parse_batch(instructions)
        counter = Counter(p.task_type for p in parsed)
        return dict(counter.most_common())
