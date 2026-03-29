"""
QuaRUM Pipeline Orchestrator.

Runs all four phases in sequence and enforces the cross-phase convergence loop.

Phase I  – Document Processing
Phase II – Knowledge Construction
Phase III– Model Generation (open → axial → selective coding)
Phase IV – UML Generation

Convergence criteria (Section 3.2.4):
  - Entity discovery rate < 5% per iteration
  - Relationship change < 10%
  - Average confidence change < 0.05
  - Stable for 2 consecutive iterations
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from checkpoints import (
    checkpoint_dir,
    load_phase1,
    load_phase3c,
    save_phase1,
    save_phase3a,
    save_phase3b,
    save_phase3c,
)
from confidence_scoring.scorer import ConfidenceScorer
from model_bundle.exporter import export_bundle
from model_bundle.schema import ModelBundle, Segment, SegmentMetadata, UMLModel
from phase1_document_processing.ingestion import ingest_document
from phase1_document_processing.cleaner import clean_text
from phase1_document_processing.segmenter import segment_text
from phase1_document_processing.metadata_attacher import attach_metadata
from phase2_knowledge_construction.embedder import Embedder
from phase2_knowledge_construction.vector_store import VectorStore
from phase2_knowledge_construction.retriever import Retriever
from phase3_model_generation.open_coding.open_coder import OpenCoder
from phase3_model_generation.axial_coding.axial_coder import AxialCoder, _normalize_and_merge
from phase3_model_generation.selective_coding.selective_coder import SelectiveCoder
from phase4_uml_generation.plantuml_generator import PlantUMLGenerator

logger = logging.getLogger(__name__)


class QuaRUMPipeline:

    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        self._output_dir = output_dir
        self._scorer = ConfidenceScorer()
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None

        os.makedirs(config.BUNDLES_DIR, exist_ok=True)
        os.makedirs(config.PLANTUML_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, file_path: str, resume_from: int = 1) -> ModelBundle:
        """
        Execute the full QuaRUM pipeline on a single requirements document.
        Returns a ModelBundle B=(M,E,C,S) ready for E-QuaRUM.

        Parameters
        ----------
        file_path   : path to the requirements document.
        resume_from : phase number to resume from (1–4).  Phases with a
                      number *lower* than resume_from are loaded from their
                      saved checkpoints instead of being re-executed.
                      1 = run everything from scratch (default).
                      2 = skip Phase I, load segments from checkpoint.
                      3 = skip Phases I–II, load vector store from checkpoint.
                      4 = skip Phases I–III, load model from checkpoint.
        """
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.info("=== QuaRUM Pipeline: %s (resume_from=%d) ===", doc_name, resume_from)

        # -----------------------------------------------------------
        # Phase I: Document Processing
        # -----------------------------------------------------------
        if resume_from <= 1:
            logger.info("Phase I: Document Processing")
            raw_text, doc_meta = ingest_document(file_path)
            cleaned_text = clean_text(raw_text, fmt=doc_meta.get("format", "txt"))
            segments = segment_text(cleaned_text, doc_meta)
            segments = attach_metadata(segments, cleaned_text)
            logger.info("  Produced %d segments", len(segments))
            save_phase1(doc_name, segments)
        else:
            segments = load_phase1(doc_name)
            logger.info("Phase I loaded from checkpoint (%d segments)", len(segments))

        # -----------------------------------------------------------
        # Phase II: Knowledge Construction
        # -----------------------------------------------------------
        if resume_from <= 2:
            logger.info("Phase II: Knowledge Construction")
            self._embedder = Embedder()
            self._vector_store = VectorStore()

            texts = [seg.text for seg in segments]
            embeddings = self._embedder.encode(texts)
            self._vector_store.build(segments, embeddings)

            vs_path = os.path.join(config.VECTOR_STORE_DIR, doc_name)
            self._vector_store.save(vs_path)

            # Also persist to the checkpoint directory so resume_from>=3 can load it
            ckpt_vs_path = str(checkpoint_dir(doc_name) / "phase2_vectorstore")
            self._vector_store.save(ckpt_vs_path)
            logger.info("  Vector store built (%d vectors)", self._vector_store.size)
        else:
            ckpt_vs_path = str(checkpoint_dir(doc_name) / "phase2_vectorstore")
            self._vector_store = VectorStore.load(ckpt_vs_path)
            self._embedder = Embedder()
            logger.info("Phase II loaded from checkpoint (%d vectors)", self._vector_store.size)

        self._retriever = Retriever(self._embedder, self._vector_store)

        # -----------------------------------------------------------
        # Phase III: Model Generation
        # -----------------------------------------------------------
        if resume_from <= 3:
            logger.info("Phase III: Model Generation")

            # Open coding
            open_coder = OpenCoder(self._retriever, self._scorer)
            entities = open_coder.run(segments)
            logger.info("  Open coding: %d entities extracted", len(entities))
            save_phase3a(doc_name, entities)

            # Axial coding (starts as soon as open coding produces entities)
            axial_coder = AxialCoder(self._retriever, self._scorer)
            relationships = axial_coder.run(entities, segments)
            logger.info("  Axial coding: %d relationships identified", len(relationships))
            save_phase3b(doc_name, relationships)

            # Selective coding (iterative until convergence)
            selective_coder = SelectiveCoder(self._retriever, self._scorer)
            entities, relationships = self._iterative_selective_coding(
                selective_coder, entities, relationships
            )
            logger.info(
                "  Selective coding converged: %d entities, %d relationships",
                len(entities), len(relationships)
            )
            # Orphan resolution: connect isolated entities or drop them
            entities, relationships = self._resolve_orphans(
                entities, relationships, segments
            )
            save_phase3c(doc_name, entities, relationships)
        else:
            entities, relationships = load_phase3c(doc_name)
            logger.info(
                "Phase III loaded from checkpoint (%d entities, %d relationships)",
                len(entities), len(relationships),
            )

        # -----------------------------------------------------------
        # Phase IV: UML Generation
        # -----------------------------------------------------------
        logger.info("Phase IV: UML Generation")
        model = UMLModel(entities=entities, relationships=relationships)
        generator = PlantUMLGenerator()
        puml_content = generator.generate(model, title=doc_name)

        puml_path = os.path.join(config.PLANTUML_DIR, f"{doc_name}.puml")
        with open(puml_path, "w", encoding="utf-8") as f:
            f.write(puml_content)
        logger.info("  PlantUML written to: %s", puml_path)

        # Optional: render to PNG using the plantuml Python package
        png_path = os.path.join(config.PLANTUML_DIR, f"{doc_name}.png")
        rendered = generator.render_to_png(puml_content, png_path)
        if rendered:
            logger.info("  PNG rendered to: %s", png_path)
        else:
            logger.info("  PNG rendering skipped (plantuml package unavailable or server unreachable)")

        # -----------------------------------------------------------
        # Build and export ModelBundle B=(M,E,C,S)
        # -----------------------------------------------------------
        bundle = self._build_bundle(
            model, entities, relationships, segments, file_path
        )
        bundle_path = os.path.join(config.BUNDLES_DIR, f"{doc_name}_bundle.json")
        export_bundle(bundle, bundle_path)
        logger.info("  Model bundle exported to: %s", bundle_path)

        return bundle

    # ------------------------------------------------------------------
    # Iterative selective coding with cross-phase convergence tracking
    # ------------------------------------------------------------------

    def _iterative_selective_coding(
        self,
        selective_coder: SelectiveCoder,
        entities: List,
        relationships: List,
    ) -> Tuple[List, List]:
        """
        Run selective coding and track convergence criteria:
          - Entity discovery rate < 5%
          - Relationship change < 10%
          - Average confidence change < 0.05
        Stable for 2 consecutive iterations.
        """
        prev_entity_count = len(entities)
        prev_rel_count = len(relationships)
        prev_avg_conf = _avg_confidence(entities, relationships)
        stable_count = 0

        for iteration in range(config.SELECTIVE_MAX_ITERATIONS):
            entities, relationships = selective_coder.run(entities, relationships)

            curr_entity_count = len(entities)
            curr_rel_count = len(relationships)
            curr_avg_conf = _avg_confidence(entities, relationships)

            entity_rate = abs(curr_entity_count - prev_entity_count) / max(1, prev_entity_count)
            rel_change = abs(curr_rel_count - prev_rel_count) / max(1, prev_rel_count)
            conf_delta = abs(curr_avg_conf - prev_avg_conf)

            logger.debug(
                "  Convergence check iter %d: entity_rate=%.3f, rel_change=%.3f, conf_delta=%.4f",
                iteration + 1, entity_rate, rel_change, conf_delta,
            )

            if (entity_rate < config.CONVERGENCE_ENTITY_RATE
                    and rel_change < config.CONVERGENCE_RELATIONSHIP_CHANGE
                    and conf_delta < config.CONVERGENCE_CONFIDENCE_DELTA):
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= config.CONVERGENCE_CONSECUTIVE_STABLE:
                logger.info(
                    "  Convergence reached after %d iterations", iteration + 1
                )
                break

            prev_entity_count = curr_entity_count
            prev_rel_count = curr_rel_count
            prev_avg_conf = curr_avg_conf

        return entities, relationships

    # ------------------------------------------------------------------
    # Orphan resolution
    # ------------------------------------------------------------------

    def _resolve_orphans(
        self,
        entities: List,
        relationships: List,
        segments: List[Segment],
    ) -> Tuple[List, List]:
        """
        After selective coding, ensure every entity participates in at least
        one relationship.  For each orphan:
          1. Find segments that mention it.
          2. Run a targeted axial coding pass on those segments (all entities).
          3. Accept any new relationships above threshold.
          4. If still orphaned, remove the entity from the model.
        """
        def _orphans(ents, rels):
            connected = {r.source for r in rels} | {r.target for r in rels}
            return [e for e in ents if e.name not in connected]

        orphan_list = _orphans(entities, relationships)
        if not orphan_list:
            return entities, relationships

        logger.info(
            "  Orphan resolution: %d isolated entities — running targeted axial pass",
            len(orphan_list),
        )

        orphan_names = {e.name.lower() for e in orphan_list}
        # Only process segments that mention at least one orphan
        orphan_segments = [
            seg for seg in segments
            if any(name in seg.text.lower() for name in orphan_names)
        ]

        if orphan_segments:
            axial_coder = AxialCoder(self._retriever, self._scorer)
            new_rels = axial_coder.run(entities, orphan_segments)
            if new_rels:
                relationships = _normalize_and_merge(relationships + new_rels)
                logger.info(
                    "  Orphan resolution added %d new relationships", len(new_rels)
                )

        # Drop entities still orphaned after the extra pass
        still_orphaned = _orphans(entities, relationships)
        if still_orphaned:
            drop_names = {e.name for e in still_orphaned}
            logger.info(
                "  Dropping %d entities still unconnected after resolution: %s",
                len(drop_names), drop_names,
            )
            entities = [e for e in entities if e.name not in drop_names]

        return entities, relationships

    # ------------------------------------------------------------------
    # Bundle construction
    # ------------------------------------------------------------------

    def _build_bundle(
        self,
        model: UMLModel,
        entities: List,
        relationships: List,
        segments: List[Segment],
        source_path: str,
    ) -> ModelBundle:
        """Assemble B = (M, E, C, S)."""

        # E: evidence mapping – element_name -> list[EvidenceSpan]
        evidence: Dict = {}
        for entity in entities:
            evidence[entity.name] = entity.evidence
            for attr in entity.attributes:
                key = f"{entity.name}.{attr.name}"
                evidence[key] = attr.evidence
            for op in entity.operations:
                key = f"{entity.name}.{op.name}"
                evidence[key] = op.evidence
        for rel in relationships:
            key = f"{rel.source}__{rel.relationship_type.value}__{rel.target}"
            evidence[key] = rel.evidence

        # C: confidence scores – element_name -> float
        confidence: Dict = {}
        for entity in entities:
            confidence[entity.name] = entity.confidence
            for attr in entity.attributes:
                confidence[f"{entity.name}.{attr.name}"] = attr.confidence
            for op in entity.operations:
                confidence[f"{entity.name}.{op.name}"] = op.confidence
        for rel in relationships:
            key = f"{rel.source}__{rel.relationship_type.value}__{rel.target}"
            confidence[key] = rel.confidence

        # S: segmentation metadata – segment_id -> SegmentMetadata
        segmentation: Dict = {seg.segment_id: seg.metadata for seg in segments}

        return ModelBundle(
            model=model,
            evidence=evidence,
            confidence=confidence,
            segmentation=segmentation,
            source_corpus=source_path,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _avg_confidence(entities: List, relationships: List) -> float:
    scores = [e.confidence for e in entities] + [r.confidence for r in relationships]
    return sum(scores) / len(scores) if scores else 0.0
