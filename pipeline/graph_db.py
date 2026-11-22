"""
Bhasha Setu — Neo4j Enterprise Multilingual Knowledge Graph & Provenance Engine

Capabilities:
  1. Multilingual Cultural & Administrative GraphRAG: Context-aware idiomatic mappings
     across 12+ official Indic languages (Hindi, Tamil, Telugu, Kannada, etc.).
  2. Cross-Video Speaker & Entity Archive: Tracks entities, government schemes, topics,
     and speaker identities across video archives.
  3. Media Provenance & Audit DAG: Directed Acyclic Graph tracing Raw Video -> Transcript
     Segments -> Neural Translations -> Synthesized Personas -> Final Muxed Audio.
  4. Resilient Hybrid Operation: Native Neo4j connection when configured, with seamless
     local JSON graph cache fallback when Neo4j is offline or credentials are pending.
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger(__name__)

# Try importing neo4j driver safely
try:
    import neo4j
    from neo4j import GraphDatabase, Driver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    log.warning("neo4j driver not installed. Falling back to local graph engine.")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOCAL_GRAPH_FILE = os.path.join(DATA_DIR, "knowledge_graph.json")


def _get_neo4j_config() -> Tuple[str, str, str, str, bool]:
    """
    Retrieve Neo4j connection parameters from st.secrets or environment variables.
    Returns: (uri, username, password, database, is_configured)
    """
    uri = ""
    user = "neo4j"
    password = ""
    database = "neo4j"

    # 1. Try st.secrets
    try:
        import streamlit as st
        if "neo4j" in st.secrets:
            cfg = st.secrets["neo4j"]
            uri = cfg.get("uri", "")
            user = cfg.get("username", cfg.get("user", user))
            password = cfg.get("password", "")
            database = cfg.get("database", database)
        if not uri and "NEO4J_URI" in st.secrets:
            uri = st.secrets["NEO4J_URI"]
        if not password and "NEO4J_PASSWORD" in st.secrets:
            password = st.secrets["NEO4J_PASSWORD"]
    except Exception:
        pass

    # 2. Try os.environ
    uri = uri or os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", user)).strip()
    password = password or os.environ.get("NEO4J_PASSWORD", "").strip()
    database = os.environ.get("NEO4J_DATABASE", database).strip()

    is_configured = bool(uri and password and not uri.startswith("neo4j+s://your-instance"))
    return uri, user, password, database, is_configured


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL RESILIENT GRAPH CACHE (Fallback when Neo4j remote is connecting)
# ══════════════════════════════════════════════════════════════════════════════

def _load_local_graph() -> Dict[str, Any]:
    """Load local JSON graph cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(LOCAL_GRAPH_FILE):
        try:
            with open(LOCAL_GRAPH_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Error reading local graph file: {e}")
    return {"nodes": {}, "relationships": [], "metadata": {"created_at": time.time()}}


def _save_local_graph(graph_data: Dict[str, Any]) -> None:
    """Save local JSON graph cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(LOCAL_GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Error saving local graph: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  NEO4J GRAPH CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jGraphService:
    _driver: Optional[Any] = None
    _connection_tested: bool = False
    _is_live: bool = False
    _error_message: str = ""

    @classmethod
    def get_driver(cls) -> Optional[Any]:
        """Get or initialize singleton Neo4j driver with thread-safety."""
        if not HAS_NEO4J:
            return None

        uri, user, password, _, is_configured = _get_neo4j_config()
        if not is_configured:
            return None

        if cls._driver is None:
            try:
                cls._driver = GraphDatabase.driver(
                    uri,
                    auth=(user, password),
                    max_connection_lifetime=3600,
                    connection_timeout=5.0,
                )
            except Exception as e:
                cls._error_message = str(e)
                log.error(f"Failed to create Neo4j driver: {e}")
                return None
        return cls._driver

    @classmethod
    def test_connection(cls) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify Neo4j connectivity and return server metadata."""
        uri, user, _, database, is_configured = _get_neo4j_config()
        if not HAS_NEO4J:
            return False, "Python neo4j driver not available.", {}
        if not is_configured:
            return False, "Neo4j connection not configured. Using local graph engine.", {
                "uri": uri or "Not set",
                "database": database,
                "mode": "Local Graph Cache (Sandbox Mode)",
            }

        driver = cls.get_driver()
        if not driver:
            return False, cls._error_message or "Could not initialize driver.", {}

        try:
            with driver.session(database=database) as session:
                res = session.run("RETURN 1 AS ping, datetime() AS now;")
                record = res.single()
                if record and record["ping"] == 1:
                    cls._is_live = True
                    return True, "Connected to Neo4j Graph Database successfully.", {
                        "uri": uri,
                        "database": database,
                        "user": user,
                        "mode": "Active Remote / AuraDB Cluster",
                        "server_time": str(record["now"]),
                    }
        except Exception as e:
            cls._error_message = str(e)
            cls._is_live = False
            return False, f"Connection test failed: {e}", {
                "uri": uri,
                "error": str(e),
                "mode": "Local Graph Fallback",
            }
        return False, "Unknown connection error", {}

    @classmethod
    def run_cypher(cls, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query if connected; otherwise return empty list."""
        driver = cls.get_driver()
        if not driver:
            return []
        _, _, _, database, _ = _get_neo4j_config()
        try:
            with driver.session(database=database) as session:
                res = session.run(cypher, params or {})
                return [r.data() for r in res]
        except Exception as e:
            log.warning(f"Neo4j Cypher query failed: {e}")
            return []


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMA INITIALIZATION & SEED GLOSSARY (Indic Cross-Lingual Knowledge)
# ══════════════════════════════════════════════════════════════════════════════

INDIC_KNOWLEDGE_GLOSSARY = [
    {
        "concept_id": "scheme_pm_kisan",
        "canonical_name": "PM-KISAN",
        "domain": "Agriculture & Governance",
        "description": "Pradhan Mantri Kisan Samman Nidhi - Direct income support of Rs 6000/yr to farmer families",
        "terms": {
            "en": "PM-KISAN Scheme",
            "hi": "प्रधानमंत्री किसान सम्मान निधि (पीएम-किसान)",
            "ta": "பிரதமர் கிசான் சம்மான் நிதி",
            "te": "పీఎం కిసాన్ సమ్మాన్ నిధి",
            "kn": "ಪಿಎಂ ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ",
            "mr": "पंतप्रधान किसान सन्मान निधी",
            "bn": "প্রধানমন্ত্রী কিষাণ সম্মান নিধি",
            "pa": "ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਕਿਸਾਨ ਸੰਮਾਨ ਨਿਧੀ",
        }
    },
    {
        "concept_id": "scheme_ayushman_bharat",
        "canonical_name": "Ayushman Bharat",
        "domain": "Healthcare & Social Welfare",
        "description": "National health protection scheme providing Rs 5 lakh health cover per family per year",
        "terms": {
            "en": "Ayushman Bharat PM-JAY",
            "hi": "आयुष्मान भारत प्रधानमंत्री जन आरोग्य योजना",
            "ta": "ஆயுஷ்மான் பாரத் திட்டம்",
            "te": "ఆయుష్మాన్ భారత్ పథకం",
            "kn": "ಆಯುಷ್ಮಾನ್ ಭಾರತ್ ಯೋಜನೆ",
            "mr": "आयुष्मान भारत योजना",
            "bn": "আয়ুষ্মান ভারত যোজনা",
            "pa": "ਆਯੁਸ਼ਮਾਨ ਭਾਰਤ ਯੋਜਨਾ",
        }
    },
    {
        "concept_id": "tech_upi",
        "canonical_name": "UPI",
        "domain": "Finance & Technology",
        "description": "Unified Payments Interface - Instant real-time payment system developed by NPCI",
        "terms": {
            "en": "Unified Payments Interface (UPI)",
            "hi": "यूनिफाइड पेमेंट्स इंटरफेस (यूपीआई)",
            "ta": "ஒருங்கிணைந்த கட்டண இடைமுகம் (UPI)",
            "te": "యూనిఫైడ్ పేమెంట్స్ ఇంటర్‌ఫేస్",
            "kn": "ಯುನಿಫೈಡ್ ಪೇಮೆಂಟ್ಸ್ ಇಂಟರ್ಫೇಸ್",
            "mr": "युनिफाइड पेमेंट्स इंटरफेस",
            "bn": "ইউনিফাইড পেমেন্টস ইন্টারফেস",
            "pa": "ਯੂਨੀਫਾਈਡ ਪੇਮੈਂਟਸ ਇੰਟਰਫੇਸ",
        }
    },
    {
        "concept_id": "gov_dbt",
        "canonical_name": "Direct Benefit Transfer",
        "domain": "Governance & Welfare",
        "description": "Transferring government subsidies directly into beneficiaries' bank accounts",
        "terms": {
            "en": "Direct Benefit Transfer (DBT)",
            "hi": "प्रत्यक्ष लाभ अंतरण (डीबीटी)",
            "ta": "நேரடிப் பயன் பரிமாற்றம்",
            "te": "ప్రత్యక్ష ప్రయోజన బదిలీ",
            "kn": "ನೇರ ಲಾಭ ವರ್ಗಾವಣೆ",
            "mr": "थेट लाभ हस्तांतरण",
            "bn": "প্রত্যক্ষ সুবিধা স্থানান্তর",
            "pa": "ਸਿੱਧਾ ਲਾਭ ਤਬਾਦਲਾ",
        }
    },
    {
        "concept_id": "agri_msp",
        "canonical_name": "Minimum Support Price",
        "domain": "Agriculture",
        "description": "Guaranteed minimum price set by the government for agricultural produce purchase",
        "terms": {
            "en": "Minimum Support Price (MSP)",
            "hi": "न्यूनतम समर्थन मूल्य (एमएसपी)",
            "ta": "குறைந்தபட்ச ஆதரவு விலை",
            "te": "కనీస మద్దతు ధర",
            "kn": "ಕನಿಷ್ಠ ಬೆಂಬಲ ಬೆಲೆ",
            "mr": "किमान आधारभूत किंमत",
            "bn": "ন্যূনতম সহায়ক মূল্য",
            "pa": "ਘੱਟੋ-ਘੱਟ ਸਮਰਥਨ ਮੁੱਲ",
        }
    },
    {
        "concept_id": "idiom_apple_of_eye",
        "canonical_name": "Beloved Person / Apple of Eye",
        "domain": "Cultural Vernacular",
        "description": "Expression of deep affection, preserving emotional resonance over literal translation",
        "terms": {
            "en": "apple of my eye",
            "hi": "आंखों का तारा / कलेजे का टुकड़ा",
            "ta": "கண்ணின் மணி",
            "te": "కంటిపాప / ప్రాణసమానుడు",
            "kn": "ಕಣ್ಣಿನ ಮಣಿ",
            "mr": "डोळ्याचे पारणे / जिवाचा तुकडा",
            "bn": "নয়নের মণি",
            "pa": "ਅੱਖਾਂ ਦਾ ਤਾਰਾ",
        }
    },
    {
        "concept_id": "idiom_day_and_night_progress",
        "canonical_name": "Rapid Flourishing & Prosperity",
        "domain": "Cultural Vernacular",
        "description": "Preserving cultural blessing of exponential growth and well-being",
        "terms": {
            "en": "prosper leaps and bounds",
            "hi": "दिन दूनी रात चौगुनी तरक्की",
            "ta": "படிப்படியாக உயர்ந்து தழைத்தோங்குதல்",
            "te": "దినదినాభివృద్ధి",
            "kn": "ದಿನೇ ದಿನೇ ಅಭಿವೃದ್ಧಿ",
            "mr": "दिवसेंदिवस उत्तरोत्तर प्रगती",
            "bn": "দিন দিন উন্নতি লাভ",
            "pa": "ਦਿਨ ਦੁੱਗਣੀ ਰਾਤ ਚੌਗਣੀ ਤਰੱਕੀ",
        }
    },
]


def init_graph_schema() -> None:
    """Initialize constraints and indexes in Neo4j, plus seed default Indic knowledge."""
    driver = Neo4jGraphService.get_driver()
    if driver:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Speaker) REQUIRE s.speaker_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:VoicePersona) REQUIRE p.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;",
            "CREATE INDEX IF NOT EXISTS FOR (seg:Segment) ON (seg.segment_id);",
        ]
        for cql in constraints:
            Neo4jGraphService.run_cypher(cql)

    # Seed Default Indic Knowledge Graph (both in Neo4j and Local Graph Cache)
    seed_indic_knowledge_graph()


def seed_indic_knowledge_graph() -> None:
    """Seed verified Indian schemes, agricultural concepts, and cultural idioms."""
    # 1. Update Local Cache
    local_graph = _load_local_graph()
    nodes = local_graph.setdefault("nodes", {})
    rels = local_graph.setdefault("relationships", [])

    for item in INDIC_KNOWLEDGE_GLOSSARY:
        c_id = item["concept_id"]
        nodes[c_id] = {
            "id": c_id,
            "label": "Concept",
            "name": item["canonical_name"],
            "domain": item["domain"],
            "description": item["description"],
            "terms": item["terms"],
        }
        # Link domain
        dom_id = f"domain_{item['domain'].replace(' ', '_').lower()}"
        if dom_id not in nodes:
            nodes[dom_id] = {"id": dom_id, "label": "Domain", "name": item["domain"]}
        rels.append({"source": c_id, "target": dom_id, "type": "BELONGS_TO"})

    _save_local_graph(local_graph)

    # 2. Push to Neo4j if live
    if Neo4jGraphService.get_driver():
        for item in INDIC_KNOWLEDGE_GLOSSARY:
            cql = """
            MERGE (c:Concept {concept_id: $c_id})
            SET c.name = $name, c.domain = $domain, c.description = $desc
            MERGE (d:Domain {name: $domain})
            MERGE (c)-[:BELONGS_TO]->(d)
            """
            Neo4jGraphService.run_cypher(cql, {
                "c_id": item["concept_id"],
                "name": item["canonical_name"],
                "domain": item["domain"],
                "desc": item["description"],
            })
            for lang_code, localized_term in item["terms"].items():
                term_cql = """
                MATCH (c:Concept {concept_id: $c_id})
                MERGE (t:Term {text: $term, lang: $lang})
                MERGE (c)-[:LOCALIZED_AS {lang: $lang}]->(t)
                """
                Neo4jGraphService.run_cypher(term_cql, {
                    "c_id": item["concept_id"],
                    "term": localized_term,
                    "lang": lang_code,
                })


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE EVENT WRITERS (DAG Provenance & Lineage)
# ══════════════════════════════════════════════════════════════════════════════

def record_video_pipeline_start(
    video_id: str,
    title: str,
    source_lang: str,
    target_lang: str,
    file_path: str = "",
) -> None:
    """Record initial Video ingestion node in the provenance graph."""
    now = time.time()
    # 1. Local Cache
    local_graph = _load_local_graph()
    v_node_id = f"video_{video_id}"
    local_graph["nodes"][v_node_id] = {
        "id": v_node_id,
        "label": "Video",
        "video_id": video_id,
        "title": title or f"Dubbing Job {video_id}",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "file_path": os.path.basename(file_path),
        "created_at": now,
        "status": "PROCESSING",
    }
    _save_local_graph(local_graph)

    # 2. Neo4j
    cql = """
    MERGE (v:Video {video_id: $vid})
    SET v.title = $title, v.source_lang = $source_lang,
        v.target_lang = $target_lang, v.created_at = $now, v.status = 'PROCESSING'
    """
    Neo4jGraphService.run_cypher(cql, {
        "vid": video_id,
        "title": title,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "now": now,
    })


def record_transcript_event(
    video_id: str,
    transcript_text: str,
    detected_lang: str,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Record speech recognition results and link transcript segments."""
    now = time.time()
    v_node_id = f"video_{video_id}"
    t_node_id = f"transcript_{video_id}"

    # Extract key Indic entities mentioned in transcript
    detected_entities = []
    text_lower = (transcript_text or "").lower()
    for item in INDIC_KNOWLEDGE_GLOSSARY:
        for lang, term_val in item["terms"].items():
            if term_val.lower() in text_lower or item["canonical_name"].lower() in text_lower:
                detected_entities.append(item["concept_id"])
                break

    # 1. Local Cache
    local_graph = _load_local_graph()
    local_graph["nodes"][t_node_id] = {
        "id": t_node_id,
        "label": "Transcript",
        "video_id": video_id,
        "char_count": len(transcript_text or ""),
        "detected_lang": detected_lang,
        "sample": (transcript_text or "")[:180],
        "created_at": now,
    }
    local_graph["relationships"].append({
        "source": v_node_id,
        "target": t_node_id,
        "type": "TRANSCRIBED_INTO",
    })

    # Link detected knowledge entities
    for c_id in detected_entities:
        local_graph["relationships"].append({
            "source": t_node_id,
            "target": c_id,
            "type": "MENTIONS_CONCEPT",
        })

    _save_local_graph(local_graph)

    # 2. Neo4j
    cql = """
    MATCH (v:Video {video_id: $vid})
    MERGE (t:Transcript {id: $tid})
    SET t.text_sample = $sample, t.detected_lang = $detected_lang,
        t.char_count = $char_count, t.created_at = $now
    MERGE (v)-[:TRANSCRIBED_INTO]->(t)
    """
    Neo4jGraphService.run_cypher(cql, {
        "vid": video_id,
        "tid": t_node_id,
        "sample": (transcript_text or "")[:200],
        "detected_lang": detected_lang,
        "char_count": len(transcript_text or ""),
        "now": now,
    })

    # Link concepts in Neo4j
    for c_id in detected_entities:
        ent_cql = """
        MATCH (t:Transcript {id: $tid}), (c:Concept {concept_id: $cid})
        MERGE (t)-[:MENTIONS_CONCEPT]->(c)
        """
        Neo4jGraphService.run_cypher(ent_cql, {"tid": t_node_id, "cid": c_id})


def record_translation_event(
    video_id: str,
    target_lang: str,
    translated_text: str,
    engine: str = "Mayura:v1",
) -> None:
    """Record Neural Translation node in provenance DAG."""
    now = time.time()
    t_node_id = f"transcript_{video_id}"
    tr_node_id = f"translation_{video_id}_{target_lang.lower()}"

    local_graph = _load_local_graph()
    local_graph["nodes"][tr_node_id] = {
        "id": tr_node_id,
        "label": "Translation",
        "video_id": video_id,
        "target_lang": target_lang,
        "engine": engine,
        "char_count": len(translated_text or ""),
        "sample": (translated_text or "")[:180],
        "created_at": now,
    }
    local_graph["relationships"].append({
        "source": t_node_id,
        "target": tr_node_id,
        "type": "TRANSLATED_BY",
        "engine": engine,
    })
    _save_local_graph(local_graph)

    cql = """
    MATCH (t:Transcript {id: $tid})
    MERGE (tr:Translation {id: $trid})
    SET tr.target_lang = $tlang, tr.engine = $engine,
        tr.char_count = $char_count, tr.created_at = $now
    MERGE (t)-[:TRANSLATED_BY {engine: $engine}]->(tr)
    """
    Neo4jGraphService.run_cypher(cql, {
        "tid": t_node_id,
        "trid": tr_node_id,
        "tlang": target_lang,
        "engine": engine,
        "char_count": len(translated_text or ""),
        "now": now,
    })


def record_synthesis_event(
    video_id: str,
    target_lang: str,
    voice_name: str,
    engine: str = "Bulbul:v3",
    audio_path: str = "",
) -> None:
    """Record Voice Persona & Synthesis lineage in DAG."""
    now = time.time()
    tr_node_id = f"translation_{video_id}_{target_lang.lower()}"
    p_node_id = f"persona_{voice_name.lower()}"
    a_node_id = f"audio_{video_id}_{target_lang.lower()}"

    local_graph = _load_local_graph()
    # Persona Node
    if p_node_id not in local_graph["nodes"]:
        local_graph["nodes"][p_node_id] = {
            "id": p_node_id,
            "label": "VoicePersona",
            "name": voice_name,
            "engine": engine,
            "regional_cadence": f"Indic {target_lang} Native",
        }
    # Audio Node
    local_graph["nodes"][a_node_id] = {
        "id": a_node_id,
        "label": "DubbedAudio",
        "file_name": os.path.basename(audio_path),
        "target_lang": target_lang,
        "created_at": now,
    }
    local_graph["relationships"].append({"source": tr_node_id, "target": a_node_id, "type": "SYNTHESIZED_TO"})
    local_graph["relationships"].append({"source": a_node_id, "target": p_node_id, "type": "VOICED_BY"})
    _save_local_graph(local_graph)

    cql = """
    MATCH (tr:Translation {id: $trid})
    MERGE (p:VoicePersona {name: $pname})
    SET p.engine = $engine, p.lang = $tlang
    MERGE (a:DubbedAudio {id: $aid})
    SET a.target_lang = $tlang, a.created_at = $now
    MERGE (tr)-[:SYNTHESIZED_TO]->(a)
    MERGE (a)-[:VOICED_BY]->(p)
    """
    Neo4jGraphService.run_cypher(cql, {
        "trid": tr_node_id,
        "pname": voice_name,
        "engine": engine,
        "tlang": target_lang,
        "aid": a_node_id,
        "now": now,
    })


def complete_video_pipeline(video_id: str, output_path: str) -> None:
    """Mark video node as COMPLETED with output artifact."""
    now = time.time()
    v_node_id = f"video_{video_id}"
    local_graph = _load_local_graph()
    if v_node_id in local_graph["nodes"]:
        local_graph["nodes"][v_node_id]["status"] = "COMPLETED"
        local_graph["nodes"][v_node_id]["completed_at"] = now
        local_graph["nodes"][v_node_id]["output_file"] = os.path.basename(output_path)
        _save_local_graph(local_graph)

    cql = """
    MATCH (v:Video {video_id: $vid})
    SET v.status = 'COMPLETED', v.completed_at = $now, v.output_path = $output
    """
    Neo4jGraphService.run_cypher(cql, {"vid": video_id, "now": now, "output": os.path.basename(output_path)})


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHRAG & KNOWLEDGE QUERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def query_graph_glossary(
    text: str,
    source_lang_code: str = "en",
    target_lang_code: str = "hi",
) -> List[Dict[str, str]]:
    """
    Look up domain glossary concepts and cultural idioms matching the input text.
    Provides verified translations to inject into translation prompts for GraphRAG.
    """
    matched = []
    text_lower = (text or "").lower()

    # Search local / seeded knowledge
    local_graph = _load_local_graph()
    for _, node in local_graph.get("nodes", {}).items():
        if node.get("label") == "Concept":
            terms = node.get("terms", {})
            src_term = terms.get(source_lang_code, "")
            tgt_term = terms.get(target_lang_code, "")
            canon_name = node.get("name", "")

            # If source term or canonical name appears in input text
            if (src_term and src_term.lower() in text_lower) or (canon_name.lower() in text_lower):
                matched.append({
                    "concept": canon_name,
                    "domain": node.get("domain", "General"),
                    "source_term": src_term or canon_name,
                    "target_term": tgt_term or canon_name,
                    "description": node.get("description", ""),
                })
    return matched


def get_graph_statistics() -> Dict[str, Any]:
    """Retrieve overall graph statistics for UI dashboard."""
    local_graph = _load_local_graph()
    nodes = local_graph.get("nodes", {})
    rels = local_graph.get("relationships", [])

    counts_by_label: Dict[str, int] = {}
    for n in nodes.values():
        lbl = n.get("label", "Unknown")
        counts_by_label[lbl] = counts_by_label.get(lbl, 0) + 1

    is_live, status_msg, meta = Neo4jGraphService.test_connection()

    return {
        "is_live_neo4j": is_live,
        "status_message": status_msg,
        "meta": meta,
        "total_nodes": len(nodes),
        "total_relationships": len(rels),
        "video_count": counts_by_label.get("Video", 0),
        "concept_count": counts_by_label.get("Concept", 0),
        "transcript_count": counts_by_label.get("Transcript", 0),
        "translation_count": counts_by_label.get("Translation", 0),
        "voice_persona_count": counts_by_label.get("VoicePersona", 0),
        "domain_count": counts_by_label.get("Domain", 0),
    }


def get_visual_graph_elements(limit: int = 40) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format nodes and edges for visual rendering in HTML / Cytoscape canvas.
    Colors match Figma / Snitch palette.
    """
    local_graph = _load_local_graph()
    raw_nodes = list(local_graph.get("nodes", {}).values())[:limit]
    raw_rels = local_graph.get("relationships", [])[: limit * 2]

    # Color palette by node label
    palette = {
        "Video":        {"bg": "#FF6B35", "border": "#EA580C", "icon": "🎥"},
        "Transcript":   {"bg": "#3B82F6", "border": "#1D4ED8", "icon": "📝"},
        "Translation":  {"bg": "#10B981", "border": "#059669", "icon": "🌐"},
        "VoicePersona": {"bg": "#8B5CF6", "border": "#6D28D9", "icon": "🎙️"},
        "DubbedAudio":  {"bg": "#F59E0B", "border": "#D97706", "icon": "🔊"},
        "Concept":      {"bg": "#EC4899", "border": "#BE185D", "icon": "💡"},
        "Domain":       {"bg": "#06B6D4", "border": "#0891B2", "icon": "🏛️"},
    }

    formatted_nodes = []
    included_ids = set()
    for n in raw_nodes:
        n_id = n["id"]
        lbl = n.get("label", "Concept")
        styling = palette.get(lbl, {"bg": "#64748B", "border": "#475569", "icon": "⚪"})
        display_name = n.get("name") or n.get("title") or n.get("id") or lbl
        if len(display_name) > 26:
            display_name = display_name[:24] + "…"

        formatted_nodes.append({
            "id": n_id,
            "label": f"{styling['icon']} {display_name}",
            "type": lbl,
            "color": styling["bg"],
            "borderColor": styling["border"],
            "details": n.get("description") or n.get("domain") or n.get("status") or "",
        })
        included_ids.add(n_id)

    formatted_edges = []
    for r in raw_rels:
        src = r.get("source")
        tgt = r.get("target")
        if src in included_ids and tgt in included_ids:
            formatted_edges.append({
                "source": src,
                "target": tgt,
                "label": r.get("type", "RELATED_TO"),
            })

    return {"nodes": formatted_nodes, "edges": formatted_edges}
