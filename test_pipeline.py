"""
Quick diagnostic test: runs each phase on the dummy Library dataset
and prints verbose output at every step.
"""

import sys, os
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('.env')

import logging
logging.basicConfig(level=logging.WARNING)   # suppress noise; we print manually

from openai import OpenAI
from phase1_document_processing.ingestion import ingest_document
from phase1_document_processing.cleaner import clean_text
from phase1_document_processing.segmenter import segment_text
from phase1_document_processing.metadata_attacher import attach_metadata
from phase2_knowledge_construction.embedder import Embedder
from phase2_knowledge_construction.vector_store import VectorStore
from phase2_knowledge_construction.retriever import Retriever
from phase3_model_generation.open_coding.prompts import OPEN_CODING_SYSTEM, OPEN_CODING_USER_TEMPLATE
from phase3_model_generation.open_coding.open_coder import _parse_json_array
from phase3_model_generation.open_coding.open_coder import OpenCoder
from phase3_model_generation.axial_coding.axial_coder import AxialCoder
from phase3_model_generation.selective_coding.selective_coder import SelectiveCoder
from phase4_uml_generation.plantuml_generator import PlantUMLGenerator
from confidence_scoring.scorer import ConfidenceScorer
import config

FILE = "dataset/LibraryTest.txt"

print("=" * 60)
print("PHASE I: Document Processing")
print("=" * 60)
raw, meta = ingest_document(FILE)
cleaned = clean_text(raw)
segments = segment_text(cleaned, meta)
segments = attach_metadata(segments, cleaned)
print(f"Segments: {len(segments)}")
for s in segments:
    print(f"  [{s.metadata.section_title!r}] tokens={s.metadata.token_count}")
    print(f"    {s.text[:100]!r}")

print()
print("=" * 60)
print("PHASE II: Knowledge Construction")
print("=" * 60)
embedder = Embedder()
vs = VectorStore()
vs.build(segments, embedder.encode([s.text for s in segments]))
retriever = Retriever(embedder, vs)
print(f"Vector store: {vs.size} vectors")

print()
print("=" * 60)
print("PHASE III-A: Open Coding (raw LLM test on first segment)")
print("=" * 60)
client = OpenAI()
seg = segments[0]
print(f"Segment: {seg.text[:200]!r}")
neighbors = retriever.retrieve_for_segment_neighbors(seg, k=3)
context = "\n---\n".join(s.text for s, _ in neighbors[:2])
msg = OPEN_CODING_USER_TEMPLATE.format(segment_text=seg.text, retrieved_context=context)
resp = client.chat.completions.create(
    model=config.LLM_MODEL,
    max_completion_tokens=config.LLM_MAX_TOKENS,
    messages=[{"role": "system", "content": OPEN_CODING_SYSTEM},
              {"role": "user", "content": msg}]
)
raw_response = resp.choices[0].message.content
print(f"\nLLM raw response:\n{raw_response}")
parsed = _parse_json_array(raw_response)
print(f"\nParsed entities: {len(parsed)}")
for e in parsed:
    print(f"  - {e.get('name')} ({e.get('element_type')}): {e.get('definition','')[:60]}")

print()
print("=" * 60)
print("PHASE III: Full Open + Axial + Selective Coding")
print("=" * 60)
scorer = ConfidenceScorer()
open_coder = OpenCoder(retriever, scorer)
entities = open_coder.run(segments)
print(f"\nOpen coding result: {len(entities)} entities")
for e in entities:
    print(f"  [{e.confidence:.3f}] {e.name} ({e.element_type.value})")
    for a in e.attributes:
        print(f"    attr: {a.name}: {a.type}")
    for op in e.operations:
        print(f"    op:   {op.name}({', '.join(op.parameters)}): {op.return_type}")

axial_coder = AxialCoder(retriever, scorer)
relationships = axial_coder.run(entities, segments)
print(f"\nAxial coding result: {len(relationships)} relationships")
for r in relationships:
    print(f"  [{r.confidence:.3f}] {r.source} -{r.relationship_type.value}-> {r.target} [{r.multiplicity_source}..{r.multiplicity_target}] role={r.role_name!r}")

selective = SelectiveCoder(retriever, scorer)
entities, relationships = selective.run(entities, relationships)
print(f"\nSelective coding result: {len(entities)} entities, {len(relationships)} relationships")

print()
print("=" * 60)
print("PHASE IV: PlantUML Generation")
print("=" * 60)
from model_bundle.schema import UMLModel
model = UMLModel(entities=entities, relationships=relationships)
gen = PlantUMLGenerator()
puml = gen.generate(model, title="LibraryTest")
print(puml)
