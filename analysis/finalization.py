"""
Finalization utilities for reviewing and confirming/rejecting hypotheses.
This module provides a Finalizer that focuses on filtering hypotheses and
checking potential false positives via LLM, without positioning it as an agent.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .agent_core import AutonomousAgent
from .concurrent_knowledge import HypothesisStore


class Finalizer(AutonomousAgent):
    """
    Specialized finalization component for hypotheses.
    Reuses graph/source utilities but focuses on targeted review.
    """

    def __init__(self, graphs_metadata_path: Path, manifest_path: Path,
                 hypothesis_path: Path, agent_id: str,
                 config: dict | None = None, debug: bool = False):
        """Initialize finalizer."""

        # Call parent init (but we'll override the model profile)
        super().__init__(graphs_metadata_path, manifest_path, agent_id, config, debug)

        # Override with finalize profile
        from llm.unified_client import UnifiedLLMClient
        self.llm = UnifiedLLMClient(
            cfg=config or {},
            profile="finalize",  # Use finalize model
            debug_logger=self.debug_logger
        )
        # Remember finalize model for reporting/metadata
        try:
            self.finalize_model = f"{self.llm.provider_name}:{self.llm.model}"
        except Exception:
            self.finalize_model = "unknown"

        # Direct access to hypothesis store for finalization
        self.hypothesis_path = hypothesis_path
        self.finalize_store = HypothesisStore(hypothesis_path, agent_id=agent_id)

        # Track what we're reviewing
        self.candidates_to_review = []
        self.review_results = []

    def finalize(self, candidates: list[tuple[str, dict]],
                 max_iterations: int = 10,
                 progress_callback: Callable[[dict], None] | None = None) -> dict:
        """
        Main finalization method - review and confirm/reject hypotheses.

        Args:
            candidates: List of (hypothesis_id, hypothesis_dict) tuples to review
            max_iterations: Max iterations for investigation
            progress_callback: Callback for progress updates

        Returns:
            Finalization report
        """
        self.candidates_to_review = candidates
        self.review_results = []

        # Record the finalization model in hypothesis store metadata for reporting
        try:
            def _set_finalize_model(data):
                meta = data.get("metadata", {})
                meta["finalization_model"] = getattr(self, 'finalize_model', 'unknown')
                data["metadata"] = meta
                return data, True
            self.finalize_store.update_atomic(_set_finalize_model)
        except Exception:
            pass

        confirmed = 0
        rejected = 0
        uncertain = 0

        for hyp_id, hypothesis in candidates:
            if progress_callback:
                progress_callback({
                    'status': 'reviewing',
                    'message': f"{hypothesis.get('title', 'Unknown')[:50]}"
                })

            # Get source files from hypothesis properties and heuristics
            source_files = list(hypothesis.get('properties', {}).get('source_files', []) or [])
            hypothesis.get('properties', {}).get('affected_functions', [])

            # Augment with file paths guessed from hypothesis text if available
            try:
                from .path_utils import guess_relpaths
                extra_texts = [
                    hypothesis.get('title', ''),
                    hypothesis.get('description', ''),
                    hypothesis.get('reasoning', ''),
                ]
                # Include evidence descriptions too
                for ev in hypothesis.get('evidence', []) or []:
                    if isinstance(ev, dict):
                        extra_texts.append(ev.get('description', '') or '')
                    elif isinstance(ev, str):
                        extra_texts.append(ev)
                guessed = guess_relpaths("\n".join([t for t in extra_texts if t]), self._repo_root)
                for rel in guessed:
                    if rel not in source_files:
                        source_files.append(rel)
            except Exception:
                pass

            # Load source code directly if available
            source_code = {}
            if source_files and self._repo_root:
                for file_path in source_files[:10]:  # Limit to 10 files
                    try:
                        full_path = self._repo_root / file_path
                        if full_path.exists():
                            with open(full_path) as f:
                                source_code[file_path] = f.read()
                    except Exception as e:
                        print(f"[!] Failed to load {file_path}: {e}")
            if progress_callback and source_code:
                try:
                    progress_callback({'status': 'info', 'message': f"Loaded {len(source_code)} file(s) from source_files"})
                except Exception:
                    pass

            # Independent repo search: augment context with likely relevant files
            if self._repo_root:
                try:
                    # Build keyword set from hypothesis (split on non-alnum + camelCase)
                    import re
                    def _kw_from_text(*texts):
                        kws = set()
                        for raw in texts:
                            s = str(raw) if raw else ''
                            for tok in re.findall(r'[A-Za-z0-9_]{3,}', s):
                                lk = tok.lower()
                                kws.add(lk)
                                subs = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', tok)
                                for sub in subs:
                                    if len(sub) >= 3:
                                        kws.add(sub.lower())
                        return kws
                    kw = _kw_from_text(
                        hypothesis.get('title', ''),
                        hypothesis.get('description', ''),
                        hypothesis.get('reasoning', ''),
                    )
                    for fn in hypothesis.get('properties', {}).get('affected_functions', []) or []:
                        if isinstance(fn, str) and fn:
                            kw.add(fn.lower())
                    exts = {'.rs','.py','.ts','.tsx','.js','.jsx','.go','.sol','.java','.kt','.swift','.c','.cc','.cpp','.h','.hpp','.yaml','.yml','.toml','.json','.sql','.sh','.rb','.php'}
                    candidates: list[tuple[int, Path]] = []
                    # Limit total files included to 10 (including any preloaded)
                    max_include = 10
                    for p in self._repo_root.rglob('*'):
                        try:
                            if not p.is_file():
                                continue
                            if p.suffix.lower() not in exts:
                                continue
                            if p.stat().st_size > 400_000:
                                continue
                            name_score = 0
                            lname = p.name.lower()
                            for t in kw:
                                if t and t in lname:
                                    name_score += 2
                            score = name_score
                            try:
                                text = p.read_text(encoding='utf-8', errors='ignore')
                                ltext = text.lower()
                                hits = sum(ltext.count(t) for t in kw if len(t) > 3)
                                score += hits
                            except Exception:
                                continue
                            if score > 0:
                                candidates.append((score, p))
                        except Exception:
                            continue
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    # Add up to the cap, skipping files already included
                    already = set(source_code.keys())
                    before = len(source_code)
                    for _, fp in candidates:
                        if len(source_code) >= max_include:
                            break
                        try:
                            rel = str(fp.relative_to(self._repo_root))
                            if rel in already:
                                continue
                            source_code[rel] = fp.read_text(encoding='utf-8', errors='ignore')
                            already.add(rel)
                        except Exception:
                            continue
                except Exception:
                    pass
                else:
                    # Only if no exception and we scanned
                    if progress_callback:
                        try:
                            added = len(source_code) - (before if 'before' in locals() else 0)
                            if added > 0:
                                progress_callback({'status': 'info', 'message': f"Repo search added {added} file(s)"})
                        except Exception:
                            pass

            # Iterative review: decide -> if uncertain and missing context, expand -> retry
            import re
            attempt = 0
            determination = None
            while True:
                attempt += 1
                review_context = self._build_review_context(hyp_id, hypothesis, source_code)
                determination = self._get_determination(review_context)
                if determination.get('verdict') in ('confirmed', 'rejected'):
                    break
                if attempt >= max_iterations:
                    break
                # Expand context based on model reasoning (filenames + identifiers), limited budget
                budget = 5
                reason_low = str(determination.get('reasoning', '')).lower()
                hints = set()
                for m in re.findall(r"[A-Za-z0-9_\-/]+\.(rs|py|ts|tsx|js|jsx|go|sol|java|kt|swift|c|cc|cpp|h|hpp|yaml|yml|toml|json|sql|sh|rb|php)", reason_low):
                    hints.add(m)
                for tok in re.findall(r"[A-Za-z0-9_]{3,}", reason_low):
                    hints.add(tok)
                    for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", tok):
                        if len(sub) >= 3:
                            hints.add(sub.lower())
                added = 0
                if self._repo_root and hints:
                    for p in self._repo_root.rglob('*'):
                        if added >= budget:
                            break
                        try:
                            if not p.is_file():
                                continue
                            if p.stat().st_size > 400_000:
                                continue
                            rel = str(p.relative_to(self._repo_root))
                            if rel in source_code:
                                continue
                            lname = rel.lower()
                            if any(h in lname for h in hints):
                                source_code[rel] = p.read_text(encoding='utf-8', errors='ignore')
                                added += 1
                        except Exception:
                            continue
                if progress_callback and added:
                    try:
                        progress_callback({'status': 'info', 'message': f"Added {added} file(s) for next iteration"})
                    except Exception:
                        pass
                if added <= 0:
                    break

            # Update hypothesis based on determination
            if determination and determination.get('verdict') == 'confirmed':
                self.finalize_store.adjust_confidence(hyp_id, 1.0, determination['reasoning'])

                # Update status and store QA comment
                def update_status(data):
                    if hyp_id in data["hypotheses"]:
                        data["hypotheses"][hyp_id]["status"] = "confirmed"
                        # Store QA comment for confirmed hypotheses
                        if determination.get('reasoning'):
                            data["hypotheses"][hyp_id]["qa_comment"] = determination['reasoning']
                        data["metadata"]["confirmed"] = sum(
                            1 for h in data["hypotheses"].values()
                            if h["status"] == "confirmed"
                        )
                    return data, True

                self.finalize_store.update_atomic(update_status)
                confirmed += 1

                if progress_callback:
                    progress_callback({
                        'status': 'confirmed',
                        'message': hypothesis.get('title', '')[:50]
                    })

                self.review_results.append({
                    'id': hyp_id,
                    'title': hypothesis.get('title'),
                    'type': hypothesis.get('vulnerability_type'),
                    'severity': hypothesis.get('severity'),
                    'verdict': 'confirmed',
                    'reasoning': determination['reasoning']
                })

            elif determination and determination.get('verdict') == 'rejected':
                self.finalize_store.adjust_confidence(hyp_id, 0.0, determination['reasoning'])

                # Update status and store QA comment
                def update_status(data):
                    if hyp_id in data["hypotheses"]:
                        data["hypotheses"][hyp_id]["status"] = "rejected"
                        # Store QA comment for rejected hypotheses
                        if determination.get('reasoning'):
                            data["hypotheses"][hyp_id]["qa_comment"] = determination['reasoning']
                    return data, True

                self.finalize_store.update_atomic(update_status)
                rejected += 1

                if progress_callback:
                    progress_callback({
                        'status': 'rejected',
                        'message': hypothesis.get('title', '')[:50]
                    })

                self.review_results.append({
                    'id': hyp_id,
                    'title': hypothesis.get('title'),
                    'verdict': 'rejected',
                    'reason': determination['reasoning']
                })

            else:
                # Uncertain - needs more investigation
                uncertain += 1
                if progress_callback:
                    progress_callback({
                        'status': 'uncertain',
                        'message': f"Needs more analysis: {hypothesis.get('title', '')[:40]}"
                    })

        # Generate report
        return self._generate_finalization_report(confirmed, rejected, uncertain)

    def _build_review_context(self, hyp_id: str, hypothesis: dict, source_code: dict[str, str]) -> str:
        """Build context for reviewing a specific hypothesis."""
        context_parts = []

        context_parts.append("=== HYPOTHESIS UNDER REVIEW ===")
        context_parts.append(f"ID: {hyp_id}")
        context_parts.append(f"Title: {hypothesis.get('title', 'Unknown')}")
        context_parts.append(f"Type: {hypothesis.get('vulnerability_type', 'unknown')}")
        context_parts.append(f"Severity: {hypothesis.get('severity', 'unknown')}")
        context_parts.append(f"Current Confidence: {hypothesis.get('confidence', 0):.0%}")
        context_parts.append(f"Status: {hypothesis.get('status', 'proposed')}")
        context_parts.append(f"Description: {hypothesis.get('description', '')}")
        context_parts.append(f"Reasoning: {hypothesis.get('reasoning', '')}")

        # Show affected functions if available
        affected_functions = hypothesis.get('properties', {}).get('affected_functions', [])
        if affected_functions:
            context_parts.append(f"Affected Functions: {', '.join(affected_functions)}")
        context_parts.append("")

        # Show evidence
        evidence = hypothesis.get('evidence', [])
        if evidence:
            context_parts.append("=== EVIDENCE ===")
            for i, e in enumerate(evidence, 1):
                e_type = e.get('type', 'unknown')
                e_desc = e.get('description', '')
                context_parts.append(f"{i}. [{e_type}] {e_desc}")
            context_parts.append("")

        # Show full source code for each referenced file
        if source_code:
            context_parts.append("=== SOURCE CODE ===")
            for file_path, code in source_code.items():
                context_parts.append(f"\n--- File: {file_path} ---")
                context_parts.append(code)
            context_parts.append("")

        return "\n".join(context_parts)

    def _get_determination(self, context: str) -> dict:
        """Get model's determination on the hypothesis."""


        try:
            # Get reasoning effort from config if available
            finalize_effort = None
            try:
                mdl_cfg = (self.config or {}).get('models', {}).get('finalize', {})
                finalize_effort = mdl_cfg.get('reasoning_effort')
            except Exception:
                pass
            
            # Request JSON and parse robustly
            response_text = self.llm.raw(
                system="You are a security expert. Respond only with valid JSON.", 
                user=context,
                reasoning_effort=finalize_effort
            )
            from utils.json_utils import extract_json_object
            obj = extract_json_object(response_text)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            print(f"[!] Failed to get determination: {e}")

        return {
            "verdict": "uncertain",
            "reasoning": (
                "After reviewing the included files, there is insufficient code context to provide a definite verdict. "
                "Load the concrete parsing/validation layers and the sink functions that enforce or perform the behavior under review."
            ),
            "confidence": 0.5
        }

    def _generate_finalization_report(self, confirmed: int, rejected: int, uncertain: int) -> dict:
        """Generate final report."""

        confirmed_vulns = [r for r in self.review_results if r.get('verdict') == 'confirmed']
        rejected_hyps = [r for r in self.review_results if r.get('verdict') == 'rejected']

        return {
            'timestamp': datetime.now().isoformat(),
            'total_reviewed': len(self.candidates_to_review),
            'confirmed': confirmed,
            'rejected': rejected,
            'uncertain': uncertain,
            'confirmed_vulnerabilities': confirmed_vulns,
            'rejected_hypotheses': rejected_hyps,
            'agent_id': self.agent_id
        }

__all__ = ['Finalizer']
