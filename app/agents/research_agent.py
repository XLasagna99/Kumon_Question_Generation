import os
import json
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.vector_store import (
    get_or_create_collection,
    register_topic,
    topic_collection_name,
)


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------
RAW_RESEARCH_DIR = os.getenv("RAW_RESEARCH_DIR") or os.getenv("RAW_RESEARCH_PATH") or "data/raw_research"
CHROMA_DIR = os.getenv("CHROMA_PATH") or "data/chroma_db"

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")

TARGET_WORDS = 10000
MAX_SEARCH_RESULTS = 12
MAX_SOURCES_FOR_DRAFT = 10
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# For Testing
TEST_TOPIC = "World War 2"

# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    """Convert a string to a URL-friendly slug.

    Args:
        text (str): The input string.

    Returns:
        str: The URL-friendly slug.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists.

    Args:
        path (Path): The path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def estimate_word_count(text: str) -> int:
    """Estimate the word count of a string.

    Args:
        text (str): The input string.

    Returns:
        int: The estimated word count.
    """
    return len(text.split())


def trim_text(text: str, max_chars: int = 12000) -> str:
    """Trim the text to a maximum number of characters.

    Args:
        text (str): The input string.
        max_chars (int, optional): The maximum number of characters. Defaults to 12000.

    Returns:
        str: The trimmed text.
    """
    return text[:max_chars]


def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate results based on their URLs.

    Args:
        results (List[Dict[str, Any]]): The list of results to deduplicate.

    Returns:
        List[Dict[str, Any]]: The deduplicated list of results.
    """
    seen = set()
    deduped = []
    for r in results:
        url = r.get("url", "").strip()
        if url and url not in seen:
            seen.add(url)
            deduped.append(r)
    return deduped


# -----------------------------
# Research Agent
# -----------------------------
class ResearchAgent:
    """AI Agent built to run research pipelines on a given topic, \
        and produce a structured JSON file of sources and notes that can \
        be used for downstream question generation and drafting.
    """
    
    def __init__(self) -> None:
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


    def build_search_queries(self, topic: str) -> List[str]:
        """
        Create a spread of queries focused on downstream question generation.
        """
        return [
            f"{topic} overview for beginners",
            f"{topic} terminology and structure",
            f"{topic} common statistics numbers measurements",
            f"{topic} realistic scenarios examples counts quantities",
            f"{topic} facts for children's educational questions",
            f"{topic} glossary key concepts",
        ]
        
        
    def search_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search for information on a given topic.

        Args:
            topic (str): The topic to search for.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        queries = self.build_search_queries(topic)
        all_results = []

        for query in queries:
            response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=4,
                include_raw_content=True,
            )
            results = response.get("results", [])
            all_results.extend(results)
            time.sleep(0.5)

        all_results = dedupe_results(all_results)
        return all_results[:MAX_SEARCH_RESULTS]
    
    
    def convert_results_to_notes(self, topic: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine researched sources into one compact notes JSON.

        The returned JSON is organized by reusable research concepts, not by
        source. Sources are kept only as a citation map and as source_ids on
        merged notes.
        """
        max_source_chars = 3600
        max_snippets_per_source = 16
        max_snippet_chars = 420
        max_summary_chars = 650
        item_limits = {
            "source_references": MAX_SEARCH_RESULTS,
            "entities": 20,
            "actions_events": 18,
            "numeric_quantities": 24,
            "realistic_ranges": 16,
            "constraints": 14,
            "question_worthy_scenarios": 20,
            "things_to_avoid": 12,
            "glossary": 20,
        }
        schema_keys = [
            "topic",
            "summary",
            "source_references",
            "entities",
            "actions_events",
            "numeric_quantities",
            "realistic_ranges",
            "constraints",
            "question_worthy_scenarios",
            "things_to_avoid",
            "glossary",
        ]
        list_keys = [key for key in schema_keys if key not in {"topic", "summary"}]
        item_keys = {
            "source_references": ["source_id", "title", "url"],
            "entities": ["name", "type", "notes", "source_ids"],
            "actions_events": ["action_or_event", "description", "source_ids"],
            "numeric_quantities": ["quantity", "value_or_range", "unit", "context", "source_ids"],
            "realistic_ranges": ["item_or_situation", "range", "reason", "source_ids"],
            "constraints": ["constraint", "why_it_matters", "source_ids"],
            "question_worthy_scenarios": [
                "scenario",
                "math_operations",
                "usable_numbers_or_ranges",
                "source_ids",
            ],
            "things_to_avoid": ["avoid", "reason", "source_ids"],
            "glossary": ["term", "definition", "source_ids"],
        }
        aliases = {
            "source_references": ["source_references", "sources", "source_refs"],
            "entities": ["entities", "people_places_things"],
            "actions_events": ["actions_events", "actions", "events"],
            "numeric_quantities": ["numeric_quantities", "quantities", "numbers"],
            "realistic_ranges": ["realistic_ranges", "ranges"],
            "constraints": ["constraints", "rules", "sanity_constraints"],
            "question_worthy_scenarios": [
                "question_worthy_scenarios",
                "scenarios",
                "arithmetic_scenarios",
                "question_scenarios",
            ],
            "things_to_avoid": ["things_to_avoid", "avoid", "bad_examples"],
            "glossary": ["glossary", "terms"],
        }
        identity_keys = {
            "source_references": ["source_id"],
            "entities": ["name", "type"],
            "actions_events": ["action_or_event"],
            "numeric_quantities": ["quantity", "unit", "context"],
            "realistic_ranges": ["item_or_situation"],
            "constraints": ["constraint"],
            "question_worthy_scenarios": ["scenario"],
            "things_to_avoid": ["avoid"],
            "glossary": ["term"],
        }
        text_limits = {
            "title": 120,
            "url": 240,
            "summary": max_summary_chars,
            "name": 70,
            "type": 35,
            "notes": 160,
            "action_or_event": 90,
            "description": 180,
            "quantity": 80,
            "value_or_range": 60,
            "unit": 35,
            "context": 160,
            "item_or_situation": 100,
            "range": 70,
            "reason": 160,
            "constraint": 120,
            "why_it_matters": 160,
            "scenario": 160,
            "avoid": 120,
            "term": 70,
            "definition": 160,
        }

        def empty_notes() -> Dict[str, Any]:
            return {
                "topic": topic,
                "summary": "",
                "source_references": [],
                "entities": [],
                "actions_events": [],
                "numeric_quantities": [],
                "realistic_ranges": [],
                "constraints": [],
                "question_worthy_scenarios": [],
                "things_to_avoid": [],
                "glossary": [],
            }

        notes = empty_notes()

        def compact_text(value: Any, limit: int) -> str:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if len(text) <= limit:
                return text
            return text[: limit - 1].rstrip() + "..."

        def parse_json_response(text: str) -> Dict[str, Any]:
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return json.loads(text)

        def coerce_source_ids(value: Any, fallback_source_id: int | None = None) -> List[int]:
            values = value if isinstance(value, list) else [value]
            source_ids = []
            for source_id in values:
                try:
                    source_id = int(source_id)
                except (TypeError, ValueError):
                    continue
                if source_id not in source_ids:
                    source_ids.append(source_id)

            if not source_ids and fallback_source_id is not None:
                source_ids.append(fallback_source_id)
            return source_ids

        def normalize_notes(candidate: Dict[str, Any], fallback_source_id: int | None = None) -> Dict[str, Any]:
            if not isinstance(candidate, dict):
                candidate = {}

            normalized = empty_notes()
            normalized["summary"] = compact_text(candidate.get("summary", ""), max_summary_chars)

            for key in list_keys:
                values = []
                for alias in aliases[key]:
                    alias_value = candidate.get(alias)
                    if isinstance(alias_value, list):
                        values.extend(alias_value)

                normalized_items = []
                for item in values:
                    if not isinstance(item, dict):
                        continue

                    normalized_item = {}
                    for item_key in item_keys[key]:
                        if item_key in item:
                            normalized_item[item_key] = item[item_key]

                    if key == "source_references":
                        normalized_item["source_id"] = (
                            normalized_item.get("source_id")
                            or item.get("id")
                            or item.get("source")
                        )
                    else:
                        source_ids = normalized_item.get("source_ids")
                        if source_ids is None:
                            source_ids = item.get("source_id") or item.get("source")
                        normalized_item["source_ids"] = coerce_source_ids(source_ids, fallback_source_id)

                    normalized_items.append(normalized_item)

                normalized[key] = normalized_items

            return normalized

        def clean_item(key: str, item: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {}
            for item_key in item_keys[key]:
                value = item.get(item_key)

                if key == "source_references" and item_key == "source_id":
                    try:
                        cleaned[item_key] = int(value)
                    except (TypeError, ValueError):
                        cleaned[item_key] = value
                elif item_key == "source_ids":
                    cleaned[item_key] = coerce_source_ids(value)
                elif item_key == "math_operations":
                    cleaned[item_key] = [
                        compact_text(operation, 24)
                        for operation in value[:4]
                    ] if isinstance(value, list) else []
                elif item_key == "usable_numbers_or_ranges":
                    cleaned[item_key] = [
                        compact_text(number, 50)
                        for number in value[:5]
                    ] if isinstance(value, list) else []
                else:
                    cleaned[item_key] = compact_text(value, text_limits.get(item_key, 120))

            return cleaned

        def identity_for(key: str, item: Dict[str, Any]) -> str:
            parts = [str(item.get(identity_key, "")).lower() for identity_key in identity_keys[key]]
            return "|".join(parts).strip("|")

        def merge_items(key: str, existing: Dict[str, Any], new_item: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)

            for item_key in item_keys[key]:
                if item_key == "source_ids":
                    merged[item_key] = sorted(set(coerce_source_ids(existing.get(item_key)) + coerce_source_ids(new_item.get(item_key))))
                elif item_key in {"math_operations", "usable_numbers_or_ranges"}:
                    values = []
                    for value in existing.get(item_key, []) + new_item.get(item_key, []):
                        if value and value not in values:
                            values.append(value)
                    merged[item_key] = values[:4 if item_key == "math_operations" else 5]
                elif not merged.get(item_key) and new_item.get(item_key):
                    merged[item_key] = new_item[item_key]
                elif new_item.get(item_key) and len(str(new_item[item_key])) > len(str(merged.get(item_key, ""))):
                    merged[item_key] = new_item[item_key]

            return merged

        def compact_and_merge_notes(candidate: Dict[str, Any], fallback_source_id: int | None = None) -> Dict[str, Any]:
            normalized = normalize_notes(candidate, fallback_source_id)
            compacted = empty_notes()
            compacted["summary"] = compact_text(normalized.get("summary", ""), max_summary_chars)

            for key in list_keys:
                merged_by_identity = {}
                for item in normalized.get(key, []):
                    item = clean_item(key, item)
                    identity = identity_for(key, item)
                    if not identity:
                        continue

                    if identity in merged_by_identity:
                        merged_by_identity[identity] = merge_items(key, merged_by_identity[identity], item)
                    else:
                        merged_by_identity[identity] = item

                compacted[key] = list(merged_by_identity.values())[: item_limits[key]]

            return compacted

        def merge_notes(base: Dict[str, Any], incoming: Dict[str, Any], fallback_source_id: int | None = None) -> Dict[str, Any]:
            merged = empty_notes()
            summaries = [
                compact_text(base.get("summary", ""), max_summary_chars),
                compact_text(incoming.get("summary", ""), max_summary_chars),
            ]
            merged["summary"] = compact_text(" ".join(summary for summary in summaries if summary), max_summary_chars)

            for key in list_keys:
                merged[key] = base.get(key, []) + incoming.get(key, [])

            return compact_and_merge_notes(merged, fallback_source_id)

        def clean_page_text(text: str) -> str:
            text = re.sub(r"\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)", " ", text)
            text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
            text = re.sub(r"\[[^\]]{0,80}\]\((?:mailto:|javascript:|sms:|#)[^)]+\)", " ", text)
            text = re.sub(r"https?://\S+", " ", text)
            text = re.sub(r"data:image/[^)\s]+", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def is_noise(text: str) -> bool:
            lowered = text.lower()
            noise_terms = [
                "cookie",
                "newsletter",
                "subscribe",
                "log in",
                "sign up",
                "share on",
                "advertisement",
                "privacy policy",
                "terms of use",
                "load comments",
            ]
            return len(text) < 35 or any(term in lowered for term in noise_terms)

        def source_to_evidence(source_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
            raw_text = result.get("raw_content") or result.get("content") or ""
            cleaned = clean_page_text(raw_text)
            topic_terms = [
                term
                for term in re.split(r"[^a-z0-9]+", topic.lower())
                if len(term) > 2
            ]
            scored_chunks = []

            for chunk in re.split(r"(?<=[.!?])\s+", cleaned):
                chunk = compact_text(chunk, max_snippet_chars)
                if is_noise(chunk):
                    continue

                lowered = chunk.lower()
                score = 0
                score += 4 if re.search(r"\d", chunk) else 0
                score += sum(1 for term in topic_terms if term in lowered)
                score += 2 if any(
                    marker in lowered
                    for marker in [
                        "average",
                        "range",
                        "total",
                        "score",
                        "points",
                        "players",
                        "teams",
                        "rules",
                        "minutes",
                        "cost",
                        "distance",
                        "weight",
                        "height",
                        "age",
                        "number",
                        "season",
                    ]
                ) else 0

                if score:
                    scored_chunks.append((score, chunk))

            snippets = []
            used_chars = 0
            seen = set()
            for _, chunk in sorted(scored_chunks, reverse=True):
                identity = chunk[:90].lower()
                if identity in seen:
                    continue
                if used_chars + len(chunk) > max_source_chars:
                    break
                snippets.append(chunk)
                used_chars += len(chunk)
                seen.add(identity)
                if len(snippets) >= max_snippets_per_source:
                    break

            if not snippets and cleaned:
                snippets.append(compact_text(cleaned, max_source_chars))

            return {
                "source_id": source_id,
                "title": compact_text(result.get("title", ""), text_limits["title"]),
                "url": compact_text(result.get("url", ""), text_limits["url"]),
                "snippets": snippets,
            }

        def add_source_reference(source: Dict[str, Any]) -> None:
            notes["source_references"].append({
                "source_id": source["source_id"],
                "title": source["title"],
                "url": source["url"],
            })

        evidence_sources = [
            source_to_evidence(i, result)
            for i, result in enumerate(results, start=1)
            if result.get("raw_content") or result.get("content")
        ]

        for source in evidence_sources:
            add_source_reference(source)
            notes = compact_and_merge_notes(notes, source["source_id"])

            if not source["snippets"]:
                continue

            prompt = f"""
            You are building one combined research-notes JSON for generating realistic
            children's arithmetic word problems.

            Topic: {topic}

            Existing combined notes JSON:
            {json.dumps(notes, ensure_ascii=False, separators=(",", ":"))}

            New source evidence:
            {json.dumps(source, ensure_ascii=False, separators=(",", ":"))}

            Task:
            Fold useful information from the new source into the existing combined notes.
            The final organization must be by concept, quantity, rule, scenario, and term.
            Do not create source-by-source summaries or source-specific buckets.
            Use source_references only as a citation map. Use source_ids on merged items.

            Keep exactly this top-level schema:
            {json.dumps(schema_keys, ensure_ascii=False)}

            Field rules:
            - entities: people, places, objects, teams, events, or concepts relevant to the topic.
            - actions_events: verbs/events that can become arithmetic situations.
            - numeric_quantities: exact numbers, units, scores, dates, counts, rates, or measures.
            - realistic_ranges: plausible values for generated questions.
            - constraints: rules or sanity checks that prevent impossible questions.
            - question_worthy_scenarios: reusable arithmetic setups, not examples tied to one article.
            - things_to_avoid: implausible or child-unsafe generation traps.
            - glossary: short definitions for useful terms.

            Size rules:
            - summary <= {max_summary_chars} characters
            - each list must stay within these caps: {json.dumps(item_limits, ensure_ascii=False)}
            - every string should be short and information-dense.

            Return valid JSON only.
            """

            try:
                response = self.openai.responses.create(
                    model=OPENAI_MODEL,
                    input=prompt,
                    max_output_tokens=2600,
                )
                incoming = compact_and_merge_notes(parse_json_response(response.output_text), source["source_id"])
                notes = merge_notes(notes, incoming, source["source_id"])
            except (json.JSONDecodeError, TypeError, AttributeError):
                notes = compact_and_merge_notes(notes, source["source_id"])

        if not evidence_sources:
            return notes

        final_prompt = f"""
        Consolidate this research-notes JSON into one globally organized JSON.

        Topic: {topic}

        Notes:
        {json.dumps(notes, ensure_ascii=False, separators=(",", ":"))}

        Requirements:
        - Return one combined JSON, not a source-by-source report.
        - Merge duplicated concepts, quantities, scenarios, constraints, and glossary terms.
        - Preserve source_references only as a citation map.
        - Keep source_ids on each non-source item.
        - Keep exactly these top-level keys: {json.dumps(schema_keys, ensure_ascii=False)}
        - Keep the same list caps: {json.dumps(item_limits, ensure_ascii=False)}
        - Keep the summary under {max_summary_chars} characters.
        - Return valid JSON only.
        """

        try:
            response = self.openai.responses.create(
                model=OPENAI_MODEL,
                input=final_prompt,
                max_output_tokens=3000,
            )
            final_notes = compact_and_merge_notes(parse_json_response(response.output_text))
            notes = merge_notes(notes, final_notes)
        except (json.JSONDecodeError, TypeError, AttributeError):
            notes = compact_and_merge_notes(notes)

        return notes

    
    def create_outline(self, topic: str, notes: Dict[str, Any]) -> str:
        notes_json = json.dumps(notes, ensure_ascii=False, indent=2)

        prompt = f"""
        Create a strong outline for a long-form research document on the topic "{topic}".

        The document's purpose is to help another AI agent generate realistic arithmetic word problems for children.
        The document should not be generic. It should emphasize:
        - terminology
        - entities
        - actions and events
        - measurable quantities
        - realistic ranges
        - scenario patterns
        - sanity constraints
        - implausible cases to avoid
        - question templates

        Use the evidence below:
        {notes_json}

        Return a numbered outline only.
        """

        response = self.openai.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return response.output_text.strip()
    
    
    def draft_document(self, topic: str, outline: str, notes: Dict[str, Any]) -> str:
        notes_json = json.dumps(notes, ensure_ascii=False, indent=2)

        prompt = f"""
        Write a long-form plaintext research document on "{topic}" for downstream AI question generation.

        Requirements:
        - Plain text only
        - Target length: about {TARGET_WORDS} words
        - Clear section headers
        - No markdown tables
        - No bullet overload
        - Must be optimized for retrieval in a vector database
        - Include a dedicated section on realistic numeric constraints
        - Include a dedicated section called "Things To Avoid"
        - Include a dedicated section called "Example Arithmetic-Friendly Scenarios"
        - Include a glossary
        - Include a short source summary section at the end

        Important:
        This document will later be embedded and retrieved by another AI agent that creates children's arithmetic questions.
        Therefore, emphasize facts that help generate semantically sensible questions.

        Outline:
        {outline}

        Evidence notes:
        {notes_json}
        """

        response = self.openai.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return response.output_text.strip()
    
    
    def quality_pass(self, topic: str, draft: str) -> str:
        prompt = f"""
        Revise the following plaintext research document on "{topic}" so that it becomes more useful for an AI system generating arithmetic word problems.

        Improve:
        - retrieval clarity
        - section naming
        - realism constraints
        - numeric usefulness
        - examples of viable scenarios
        - avoidance of vague filler

        Keep it as plaintext.
        Do not use markdown tables.
        Preserve and improve length where helpful.

        Document:
        \"\"\"
        {draft}
        \"\"\"
        """

        response = self.openai.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return response.output_text.strip()
    
    
    def save_outputs(
        self,
        topic: str,
        search_results: List[Dict[str, Any]],
        notes: Dict[str, Any],
        final_text: str,
    ) -> Path:
        topic_slug = slugify(topic)
        topic_dir = Path(RAW_RESEARCH_DIR) / topic_slug
        ensure_dir(topic_dir)

        sources_path = topic_dir / "sources.json"
        notes_path = topic_dir / "extracted_notes.json"
        final_txt_path = topic_dir / "final_research.txt"

        with open(sources_path, "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return final_txt_path
    
    
    def ingest_into_vector_store(self, topic: str, txt_path: Path) -> None:
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Research text file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            raise ValueError(f"Research text file is empty: {txt_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_text(text)
        collection_name = topic_collection_name(topic)
        collection = get_or_create_collection(
            collection_name,
            metadata={"topic": topic},
        )

        path_hash = hashlib.sha1(str(txt_path.resolve()).encode("utf-8")).hexdigest()[:12]
        ids = [
            f"{collection_name}_{path_hash}_{index}"
            for index in range(len(chunks))
        ]
        metadatas = [
            {
                "topic": topic,
                "source_type": "research_txt",
                "file_name": txt_path.name,
                "path": str(txt_path),
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        register_topic(topic, collection_name, len(chunks))
        
        
    def run(self, topic: str) -> Dict[str, Any]:
        # Run Tavily serrch to gather sources and remove duplicates from results
        search_results = self.search_topic(topic)
        notes = self.convert_results_to_notes(topic, search_results)
        outline = self.create_outline(topic, notes)
        draft = self.draft_document(topic, outline, notes)
        final_text = self.quality_pass(topic, draft)

        txt_path = self.save_outputs(topic, search_results, notes, final_text)
        self.ingest_into_vector_store(topic, txt_path)

        return {
            "topic": topic,
            "txt_path": str(txt_path),
            "word_count": estimate_word_count(final_text),
            "num_sources": len(search_results),
            "num_notes": sum(
                len(notes.get(key, []))
                for key in [
                    "entities",
                    "actions_events",
                    "numeric_quantities",
                    "realistic_ranges",
                    "constraints",
                    "question_worthy_scenarios",
                    "things_to_avoid",
                    "glossary",
                ]
            ),
        }


if __name__ == "__main__":
    topic = TEST_TOPIC
    agent = ResearchAgent()
    result = agent.run(topic)
    print(json.dumps(result, indent=2))
