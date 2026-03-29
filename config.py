
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-5-nano"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 16000         # gpt-5-nano is a reasoning model; needs large budget for reasoning + output
LLM_PARALLEL_WORKERS = 5       # concurrent LLM calls (open coding: segments; axial coding: pairs)

# ---------------------------------------------------------------------------
# Embedding  (all-MiniLM-L6-v2, no fine-tuning)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Phase I – Segmentation
# Target: 350-600 tokens; cap: 800; overlap: 50 tokens
# (RecursiveCharacterTextSplitter uses characters; 1 token ≈ 4 chars)
# ---------------------------------------------------------------------------
SEGMENT_TARGET_MIN_TOKENS = 350
SEGMENT_TARGET_MAX_TOKENS = 600
SEGMENT_CAP_TOKENS = 800
SEGMENT_CHUNK_SIZE = 512        # chars, matches paper implementation table
SEGMENT_OVERLAP = 50            # chars, matches paper implementation table
SEGMENT_MIN_TOKENS = 30         # discard very small segments

# ---------------------------------------------------------------------------
# Phase II – Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 5             # 3–5; paper empirically validated; use 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.0   # cosine; FAISS returns ranked list

# ---------------------------------------------------------------------------
# Confidence scoring – thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70     # τ – accept threshold for all phases

# Open coding weights: score_open = clip(0.50E + 0.30F + 0.15R + 0.05C)
SCORE_OPEN_E = 0.50
SCORE_OPEN_F = 0.30
SCORE_OPEN_R = 0.15
SCORE_OPEN_C = 0.05

# Axial coding weights: score_axial = clip(0.35E + 0.35F + 0.15R + 0.05C + 0.10S)
SCORE_AXIAL_E = 0.35
SCORE_AXIAL_F = 0.35
SCORE_AXIAL_R = 0.15
SCORE_AXIAL_C = 0.05
SCORE_AXIAL_S = 0.10

# Selective coding weights: score_sel = clip(0.70*score_prior + 0.30*U)
SCORE_SEL_PRIOR = 0.70
SCORE_SEL_UML = 0.30

# Fixed score for anchor-based alignment (no fine-grained offsets)
ANCHOR_ALIGNMENT_SCORE = 0.85

# ---------------------------------------------------------------------------
# Phase III – Open coding
# ---------------------------------------------------------------------------
ENTITY_MERGE_NAME_SIMILARITY = 0.85    # merge duplicates above this
OPEN_CODING_REQUERY_ON_LOW_CONFIDENCE = True

# ---------------------------------------------------------------------------
# Phase III – Axial coding guardrails
# ---------------------------------------------------------------------------
AXIAL_MIN_ENTITY_CONFIDENCE = 0.70     # do not trigger axial on entities below this
AXIAL_MAX_CHECKS_PER_SEGMENT = 20      # cap axial checks per segment
AXIAL_PAUSE_SCORE_THRESHOLD = 0.50     # pause pair after 3 consecutive below this
AXIAL_PAUSE_CONSECUTIVE = 3

# Relationship type priority order (most specific first)
RELATIONSHIP_TYPE_PRIORITY = [
    "IS_A",
    "IMPLEMENTS",
    "IS_PART_OF",
    "AGGREGATES",
    "ASSOCIATES",
    "DEPENDS_ON",
]

# ---------------------------------------------------------------------------
# Phase III – Selective coding convergence
# ---------------------------------------------------------------------------
CONVERGENCE_ENTITY_RATE = 0.05         # <5% new entities per iteration
CONVERGENCE_RELATIONSHIP_CHANGE = 0.10 # <10% relationships change
CONVERGENCE_CONFIDENCE_DELTA = 0.05    # avg confidence change < 0.05
CONVERGENCE_CONSECUTIVE_STABLE = 2     # stable for 2 consecutive iterations
SELECTIVE_MAX_ITERATIONS = 10          # safety cap

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
BUNDLES_DIR = "output/bundles"
PLANTUML_DIR = "output/plantuml"
LOGS_DIR = "output/logs"
VECTOR_STORE_DIR = "output/vector_store"
