#!/usr/bin/env python3
"""Dynamic Knowledge Graph Builder with agent-driven schema discovery."""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import LLMClient
from pydantic import BaseModel, Field


@dataclass
class DynamicNode:
    """Flexible node representation for dynamic graphs."""
    id: str
    type: str  # Dynamic type decided by agent
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional structured fields
    description: Optional[str] = None
    confidence: float = 1.0
    source_refs: List[str] = field(default_factory=list)  # File paths or card IDs
    created_by: str = "agent"  # Which agent/pass created this
    iteration: int = 0  # When in the process it was created
    
    # Analysis fields - facts about the system (NOT security issues)
    observations: List[Dict[str, Any]] = field(default_factory=list)  # Verified facts, invariants, behaviors
    assumptions: List[Dict[str, Any]] = field(default_factory=list)  # Unverified assumptions, constraints


@dataclass
class DynamicEdge:
    """Flexible edge representation for dynamic graphs."""
    id: str
    type: str  # Dynamic type decided by agent
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Optional structured fields
    label: Optional[str] = None
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)  # Supporting evidence
    created_by: str = "agent"
    iteration: int = 0


@dataclass
class KnowledgeGraph:
    """A single knowledge graph with a specific focus"""
    name: str
    focus: str  # What this graph focuses on (structure, security, data flow, etc.)
    nodes: Dict[str, DynamicNode] = field(default_factory=dict)
    edges: Dict[str, DynamicEdge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: DynamicNode):
        self.nodes[node.id] = node
    
    def add_edge(self, edge: DynamicEdge):
        self.edges[edge.id] = edge
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighboring nodes, optionally filtered by edge type"""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if edge_type is None or edge.type == edge_type:
                    neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                if edge_type is None or edge.type == edge_type:
                    neighbors.append(edge.source_id)
        return neighbors



class GraphSpec(BaseModel):
    """Specification for a graph to build"""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Graph name")
    focus: str = Field(description="What this graph focuses on")


class GraphDiscovery(BaseModel):
    """Initial discovery of what graphs to build"""
    model_config = {"extra": "forbid"}
    graphs_needed: List[GraphSpec] = Field(
        default_factory=list,
        description="List of graphs to create"
    )
    suggested_node_types: List[str] = Field(
        default_factory=list,
        description="Custom node types needed for this codebase"
    )
    suggested_edge_types: List[str] = Field(
        default_factory=list,
        description="Custom edge types needed for this codebase"
    )




class Observation(BaseModel):
    """An observation or verified fact about a node (NOT security issues - use hypotheses for those)"""
    model_config = {"extra": "forbid"}
    description: str = Field(description="Description of the observation")
    type: str = Field(default="general", description="Type: invariant, behavior, pattern, constraint, property")
    confidence: float = Field(1.0, description="Confidence in this observation")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this observation")


class Assumption(BaseModel):
    """An unverified assumption about a node"""
    model_config = {"extra": "forbid"}
    description: str = Field(description="Description of the assumption")
    type: str = Field(default="general", description="Type: constraint, precondition, invariant, etc.")
    confidence: float = Field(0.5, description="Confidence in this assumption")
    needs_verification: bool = Field(True, description="Whether this needs verification")


class NodeSpec(BaseModel):
    """Node to add to the graph"""
    model_config = {"extra": "forbid"}
    id: str = Field(description="Unique node identifier (e.g., 'func_calculate', 'module_utils')")
    type: str = Field(description="Node type (e.g., function, class, module)")
    label: str = Field(description="Human-readable label for the node")
    refs: List[str] = Field(default_factory=list, description="List of card IDs where this node appears")


class EdgeSpec(BaseModel):
    """Edge to add - connects two nodes"""
    model_config = {"extra": "forbid"}
    type: str = Field(description="Edge type (e.g., calls, uses, depends_on)")
    src: str = Field(description="Source node ID (must be an existing node ID, NOT a card ID)")
    dst: str = Field(description="Target node ID (must be an existing node ID, NOT a card ID)")
    refs: List[str] = Field(default_factory=list, description="Card IDs that evidence this edge")


class NodeUpdate(BaseModel):
    """Update for an existing node"""
    model_config = {"extra": "forbid"}
    id: str = Field(description="Node ID to update")
    description: Optional[str] = Field(None, description="New description")
    properties: Optional[str] = Field(None, description="JSON string of properties to update")
    
    # New observations/assumptions to add
    new_observations: List[Observation] = Field(default_factory=list, description="[LEAVE EMPTY during graph building - only for agent analysis phase]")
    new_assumptions: List[Assumption] = Field(default_factory=list, description="[LEAVE EMPTY during graph building - only for agent analysis phase]")


class GraphUpdate(BaseModel):
    """Incremental update to a knowledge graph"""
    model_config = {"extra": "forbid"}
    target_graph: str = Field(default="", description="Name of graph to update")
    
    new_nodes: List[NodeSpec] = Field(
        default_factory=list,
        description="New nodes to add - return empty list if no new nodes found"
    )
    new_edges: List[EdgeSpec] = Field(
        default_factory=list,
        description="New edges to add - return empty list if no new edges found"
    )
    
    node_updates: List[NodeUpdate] = Field(
        default_factory=list,
        description="Updates to existing nodes with new invariants/observations"
    )





class GraphBuilder:
    """
    Agent-driven dynamic knowledge graph builder.
    
    Key principles:
    1. Agent decides what's important to model
    2. Multiple specialized graphs for different concerns
    3. Iterative refinement based on discoveries
    4. Minimal pre-processing - just code cards
    """
    
    def __init__(self, config: Dict, debug: bool = False):
        self.config = config
        self.debug = debug
        
        # Initialize LLM clients - guidance for discovery, graph for building
        # Graph model for building graphs
        self.llm = LLMClient(config, profile="graph")
        if debug:
            graph_model = config.get("models", {}).get("graph", {}).get("model", "unknown")
            print(f"[*] Graph model: {graph_model}")
        
        # Guidance model for initial discovery (heavier reasoning)
        self.llm_agent = LLMClient(config, profile="guidance")
        if debug:
            agent_model = config.get("models", {}).get("guidance", {}).get("model", "unknown")
            print(f"[*] Agent model: {agent_model} (for discovery)")
        
        # Knowledge graphs storage
        self.graphs: Dict[str, KnowledgeGraph] = {}
        
        # Card storage for later retrieval
        self.card_store: Dict[str, Dict] = {}
        
        # Iteration counter
        self.iteration = 0
        # External progress sink
        self._progress_callback = None

    def _emit(self, status: str, message: str, **kwargs):
        """Emit progress events to callback and optionally print when debug."""
        if self._progress_callback:
            payload = {"status": status, "message": message, "iteration": self.iteration}
            payload.update(kwargs)
            try:
                # Prefer dict signature
                self._progress_callback(payload)
            except TypeError:
                # Backward compatibility: (iteration, message)
                try:
                    self._progress_callback(self.iteration, message)
                except Exception:
                    pass
        if self.debug:
            print(f"[{status}] {message}")
    
    def build(
        self,
        manifest_dir: Path,
        output_dir: Path,
        max_iterations: int = 5,
        focus_areas: Optional[List[str]] = None,
        max_graphs: int = 2,
        force_graphs: Optional[List[Dict[str, str]]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for dynamic graph building.
        
        Args:
            manifest_dir: Directory with code manifest and cards
            output_dir: Where to save results
            max_iterations: Maximum refinement iterations
            focus_areas: Optional list of areas to focus on
            max_graphs: Maximum number of graphs to create
            force_graphs: Optional list of specific graphs to force (e.g., [{"name": "CallGraph", "focus": "function calls"}])
        """
        start_time = time.time()
        self._progress_callback = progress_callback
        
        # Load code cards (minimal preprocessing)
        manifest, cards = self._load_manifest(manifest_dir)
        
        # Calculate statistics
        total_chars = sum(len(card.get("content", "")) + 
                         len(card.get("peek_head", "")) + 
                         len(card.get("peek_tail", "")) 
                         for card in cards)
        # Estimate lines from peek content
        total_lines = sum(
            card.get("peek_head", "").count('\n') + 
            card.get("peek_tail", "").count('\n') + 2
            for card in cards
        )
        
        self._emit("start", "Dynamic Graph Building")
        self._emit("stats", f"Files: {manifest['num_files']}")
        self._emit("stats", f"Cards: {len(cards)}")
        self._emit("stats", f"Total lines: {total_lines:,}")
        self._emit("stats", f"Total chars: {total_chars:,}")
        self._emit("stats", f"Max iterations: {max_iterations}")
        
        # Phase 1: Discovery - Let agent decide what to build
        self._emit("phase", "Graph Discovery")
        self._discover_graphs(manifest, cards, focus_areas, max_graphs, force_graphs)
        
        # Phase 2: Iterative Graph Building
        if self.debug:
            print("\n[Phase 2] Iterative Graph Building")
        for i in range(max_iterations):
            self.iteration = i
            if self.debug:
                print(f"  Iteration {i+1}/{max_iterations}")
            self._emit("building", f"Building graphs: iteration {i+1}/{max_iterations}")
            
            # Build/refine graphs (no early stopping)
            self._build_iteration(cards)
        
        # Phase 3: Save results
        self._emit("phase", "Saving Results")
        results = self._save_results(output_dir, manifest)
        
        duration = time.time() - start_time
        self._emit("complete", f"Complete in {duration:.1f}s")
        
        return results
    
    def _discover_graphs(
        self,
        manifest: Dict,
        cards: List[Dict],
        focus_areas: Optional[List[str]] = None,
        max_graphs: int = 2,
        force_graphs: Optional[List[Dict[str, str]]] = None
    ):
        """Let agent discover what graphs to build"""
        
        # If specific graphs are forced, use them directly
        if force_graphs:
            self._emit("discover", "Using forced graph specifications...")
            for graph_spec in force_graphs:
                name = graph_spec["name"].replace(' ', '_').replace('/', '_')
                focus = graph_spec["focus"]
                self.graphs[name] = KnowledgeGraph(
                    name=name,
                    focus=focus,
                    metadata={"created_at": time.time(), "display_name": graph_spec["name"]}
                )
                self._emit("graph", f"Created graph: {graph_spec['name']} (focus: {focus})")
            return
        
        # Use adaptive sampling based on content size, not count
        self._emit("discover", "Analyzing codebase for graph discovery...")
        # Auto-sample based on content size to stay within context limits
        code_samples = self._sample_cards(cards, target_size_mb=2.0)
        
        # Allow forcing specific graph type through focus_areas (backward compatibility)
        if focus_areas and "call_graph" in focus_areas:
            system_prompt = f"""Create {max_graphs} call graph(s) showing function/method calls.
For each graph, provide:
- name: A short name for the graph (e.g., "CallGraph")
- focus: What this graph focuses on (e.g., "function call relationships")"""
        else:
            system_prompt = f"""Design EXACTLY {max_graphs} graph{'s' if max_graphs > 1 else ''} for this codebase.

REQUIRED: The FIRST graph MUST be a high-level system/component/flow graph that shows:
- Major components, modules, or contracts
- How they relate and interact
- Data/control flow between them
- System boundaries and external interfaces

Name it "SystemArchitecture".

{'For additional graphs, choose what would be most useful for understanding the system architecture and relationships in this codebase.' if max_graphs > 1 else ''}

{'Examples of other useful graphs:' if max_graphs > 1 else ''}
{'''- Call graph: function calls
- Authorization / action map
- Data flow graph
- State mutation graph
- User flows and evens
- Asset/monetary flows (for smart contract projects)''' if max_graphs > 1 else ''}

For each graph, you MUST provide:
- name: A short name for the graph
- focus: What this graph focuses on (be specific)

IMPORTANT: Return EXACTLY {max_graphs} graph{'s' if max_graphs > 1 else ''}, no more, no less.
The FIRST must be the system/component/flow overview."""
        
        user_prompt = {
            "repository": manifest.get("repo_path", "unknown"),
            "num_files": manifest["num_files"],
            "code_samples": code_samples,
            "focus_areas": focus_areas or ["general analysis"],
            "instruction": "Determine what knowledge graphs to build and what custom types are needed"
        }
        
        # Use agent model for discovery (better reasoning)
        discovery = self.llm_agent.parse(
            system=system_prompt,
            user=json.dumps(user_prompt, indent=2),
            schema=GraphDiscovery
        )
        
        # Create the suggested graphs (limited to max_graphs)
        graphs_to_create = discovery.graphs_needed[:max_graphs]
        if len(discovery.graphs_needed) > max_graphs:
            self._emit("discover", f"LLM suggested {len(discovery.graphs_needed)} graphs, limiting to {max_graphs}")
        
        for i, graph_spec in enumerate(graphs_to_create):
            raw_name = graph_spec.name if hasattr(graph_spec, 'name') else graph_spec.get("name", f"graph_{len(self.graphs)}")
            focus = graph_spec.focus if hasattr(graph_spec, 'focus') else graph_spec.get("focus", "general")
            
            # Force first graph to be SystemArchitecture
            if i == 0 and len(self.graphs) == 0:
                raw_name = "SystemArchitecture"
            
            # Sanitize name for file system (replace spaces with underscores)
            name = raw_name.replace(' ', '_').replace('/', '_')
            
            self.graphs[name] = KnowledgeGraph(
                
                name=name,
                focus=focus,
                metadata={"created_at": time.time(), "display_name": raw_name}
            )
            self._emit("graph", f"Created graph: {raw_name} (focus: {focus})")
        
        # Note custom types (agent can use these later)
        if discovery.suggested_node_types:
            self._emit("note", f"Custom node types: {', '.join(discovery.suggested_node_types)}")
        if discovery.suggested_edge_types:
            self._emit("note", f"Custom edge types: {', '.join(discovery.suggested_edge_types)}")
    
    def _build_iteration(self, cards: List[Dict]):
        """
        Single iteration to build/refine graphs.
        Continues through all iterations without early stopping.
        """
        
        # Always try to use ALL cards for maximum context
        # The model needs to see the entire codebase to make good decisions
        for graph_name, graph in self.graphs.items():
            orphan_count = len(self._get_orphaned_nodes(graph))
            self._emit("graph_build", f"{graph_name}: {len(graph.nodes)}N/{len(graph.edges)}E, {orphan_count} orphans")
            
            # Use adaptive sampling based on content size to respect context limits
            relevant_cards = self._sample_cards(cards, target_size_mb=2.0)
            if len(relevant_cards) != len(cards):
                self._emit("sample", f"Sampled {len(relevant_cards)} cards from {len(cards)} total for context limits")
            
            # Update the graph
            update = self._update_graph(graph, relevant_cards)
            
            if update:
                # Apply whatever updates were found (could be empty lists)
                added_nodes = len(update.new_nodes)
                added_edges = len(update.new_edges)
                self._apply_update(graph, update)
                
                new_orphan_count = len(self._get_orphaned_nodes(graph))
                if added_nodes > 0 or added_edges > 0:
                    self._emit("update", f"Added: {added_nodes} nodes, {added_edges} edges (orphans {orphan_count}->{new_orphan_count})")
                else:
                    self._emit("update", f"No new nodes/edges (orphans: {new_orphan_count})")
    
    def _get_orphaned_nodes(self, graph: KnowledgeGraph) -> set:
        """Find nodes with no edges (neither incoming nor outgoing)"""
        connected_nodes = set()
        for edge in graph.edges.values():
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)
        
        orphaned = set(graph.nodes.keys()) - connected_nodes
        return orphaned
    
    def _update_graph(self, graph: KnowledgeGraph, cards: List[Dict]) -> Optional[GraphUpdate]:
        """Update graph with new nodes and edges based on current state"""
        
        # Store cards for later retrieval
        cards_with_ids = []
        for card in cards:
            card_id = card.get("id", f"card_{len(self.card_store)}_{len(cards_with_ids)}")
            self.card_store[card_id] = card
            card_with_id = dict(card)
            card_with_id["id"] = card_id
            cards_with_ids.append(card_with_id)
        
        # Adaptive prompting based on graph state
        if self.iteration == 0:
            # Initial build - focus on discovering CONNECTED nodes
            system_prompt = f"""Build {graph.focus} graph.
FOCUS: Find core nodes AND their connections at fine granularity.

IMPORTANT: This is ONLY structural discovery - do NOT add observations or assumptions.
Those will be added later during analysis.

CRITICAL: Only include nodes for code that EXISTS in this codebase's source files.
DO NOT create nodes for:
- External dependencies (OpenZeppelin, Chainlink, etc.)
- Standard library contracts or interfaces imported from outside
- Third-party libraries
Only reference external dependencies in edge relationships if needed.

Nodes: id (unique string), type, label, refs (array of card IDs that contain this node)
  - Prefer function-level and storage-level nodes; contract-level nodes are acceptable but should not crowd out finer nodes.
  - MUST be defined in the project's source files (not just imported/used)
Edges: type, src (source NODE ID), dst (target NODE ID), refs (array of card IDs evidencing this relationship)

CRITICAL - refs field:
- Each node MUST have a refs array containing the IDs of cards where this node appears
- Each edge SHOULD have a refs array with card IDs where this relationship is visible (call site, data flow, state mutation)
- Look at the code_samples - each has an "id" field like "card_0_0", "card_0_1", etc.
- Include these card IDs in refs for nodes/edges found in those cards
- Example: if you find a function in card_0_0 and card_0_3, refs should be ["card_0_0", "card_0_3"]

IMPORTANT: 
- Edge src/dst must reference node IDs you created, NOT card IDs!
- Every node should have at least one edge (incoming or outgoing)
- Prioritize connected components over isolated nodes
Target: 15-25 nodes with strong connectivity."""
        else:
            # Refinement - strongly prioritize connecting existing nodes
            orphaned_nodes = self._get_orphaned_nodes(graph)
            orphan_count = len(orphaned_nodes)
            
            if orphan_count > 5:
                # Many orphaned nodes - focus on connecting them
                orphan_sample = list(orphaned_nodes)[:10]  # Show first 10
                focus_instruction = f"CRITICAL: {orphan_count} nodes have NO connections! Connect these orphans: {orphan_sample}"
            elif len(graph.edges) < len(graph.nodes):
                focus_instruction = f"PRIORITY: Find EDGES for existing {len(graph.nodes)} nodes. Each node needs connections!"
            else:
                focus_instruction = "Balance nodes and edges. Ensure all nodes are connected."
            
            system_prompt = f"""Refine {graph.focus}. Current: {len(graph.nodes)}N/{len(graph.edges)}E, {orphan_count} orphans.
{focus_instruction}

IMPORTANT: This is ONLY structural discovery - do NOT add observations or assumptions.
Those will be added later during analysis.

CRITICAL: Only include nodes for code that EXISTS in this codebase's source files.
DO NOT create nodes for:
- External dependencies (OpenZeppelin, Chainlink, etc.)
- Standard library contracts or interfaces imported from outside
- Third-party libraries
Only reference external dependencies in edge relationships if needed.

DO NOT add new nodes unless they connect to existing ones.
Nodes: If adding new nodes, include refs array with card IDs where they appear; prefer function/storage granularity.
  - MUST be defined in the project's source files (not just imported/used)
Edges: type, src (existing NODE id), dst (existing NODE id), refs (card IDs evidencing the relationship). 
IMPORTANT: Use existing node IDs! Check the existing_nodes list.
For any new nodes, include refs field with card IDs from code_samples.
Return empty lists only if graph is fully connected."""
        
        # Build user prompt with existing nodes for reference
        existing_nodes_list = []
        if self.iteration > 0 and len(graph.nodes) > 0:
            # Provide existing nodes to help model make connections
            for node_id, node in graph.nodes.items():
                existing_nodes_list.append({
                    "id": node_id,
                    "type": node.type,
                    "label": node.label
                })
        
        user_prompt = {
            "graph_name": graph.name,
            "graph_focus": graph.focus,
            "existing_nodes": existing_nodes_list if existing_nodes_list else f"{len(graph.nodes)} nodes",
            "existing_edges": len(graph.edges),
            "code_samples": cards_with_ids,
            "iteration": self.iteration,
            "instruction": "Update graph. Use NODE IDs for edges. Include refs arrays for nodes AND edges with card IDs from code_samples that evidence them."
        }
        
        try:
            update = self.llm.parse(
                system=system_prompt,
                user=json.dumps(user_prompt, indent=2),
                schema=GraphUpdate
            )
            update.target_graph = graph.name
            return update
        except Exception as e:
            self._emit("warn", f"Failed to get update: {e}")
            return None
    
    
    def _apply_update(self, graph: KnowledgeGraph, update: GraphUpdate):
        """Apply an update to a graph"""
        
        # Add new nodes
        for node_spec in update.new_nodes:
            # Parse properties if provided as JSON string
            properties = {}
            
            node = DynamicNode(
                id=node_spec.id,
                type=node_spec.type,
                label=node_spec.label,
                properties=properties,
                description=None,  # No description in compact schema
                confidence=1.0,  # Default confidence
                source_refs=node_spec.refs,  # Use shortened field name
                created_by=f"iteration_{self.iteration}",
                iteration=self.iteration,
                # Empty security fields since we're focusing on structure
                observations=[],
                assumptions=[]
            )
            graph.add_node(node)
        
        # Add new edges
        for edge_spec in update.new_edges:
            edge = DynamicEdge(
                id=self._generate_id("edge"),
                type=edge_spec.type,
                source_id=edge_spec.src,  # Use shortened field name
                target_id=edge_spec.dst,  # Use shortened field name
                properties={},
                label=edge_spec.type,  # Use type as label
                confidence=1.0,
                evidence=edge_spec.refs or [],
                created_by=f"iteration_{self.iteration}",
                iteration=self.iteration
            )
            graph.add_edge(edge)
        
        # Update existing nodes
        for node_update in update.node_updates:
            if node_update.id in graph.nodes:
                node = graph.nodes[node_update.id]
                if node_update.description:
                    node.description = node_update.description
                if node_update.properties:
                    try:
                        props = json.loads(node_update.properties)
                        node.properties.update(props)
                    except json.JSONDecodeError:
                        pass
                
                # Add new observations and assumptions
                for obs in node_update.new_observations:
                    node.observations.append(obs.model_dump())
                for assum in node_update.new_assumptions:
                    node.assumptions.append(assum.model_dump())
        
        # Updates complete
    
    
    def _sample_cards(
        self,
        cards: List[Dict],
        target_size_mb: float = 2.0
    ) -> List[Dict]:
        """Adaptive sampling based on content size instead of count"""
        
        # Calculate total size of cards
        total_size = sum(
            len(card.get("content", "")) + 
            len(card.get("peek_head", "")) + 
            len(card.get("peek_tail", ""))
            for card in cards
        )
        
        # Convert MB to bytes
        size_threshold = int(target_size_mb * 1_000_000)
        
        # If under threshold, use all cards
        if total_size <= size_threshold:
            if self.debug:
                print(f"      Using ALL {len(cards)} cards (total size: {total_size:,} chars < {size_threshold:,} threshold)")
            return cards
        
        # Calculate proportional sample size to stay under threshold
        ratio = size_threshold / total_size
        sample_size = max(1, int(len(cards) * ratio))
        
        if self.debug:
            print(f"      WARNING: Large codebase - sampling {sample_size} cards from {len(cards)} total (size: {total_size:,} chars > {size_threshold:,} threshold, ratio: {ratio:.3f})")
        
        # Use original file diversity sampling logic
        import random
        
        # Try to sample from different files if possible
        files_to_cards = {}
        for card in cards:
            file_path = card.get("relpath", "unknown")
            if file_path not in files_to_cards:
                files_to_cards[file_path] = []
            files_to_cards[file_path].append(card)
        
        # Sample cards ensuring file diversity
        sampled = []
        files = list(files_to_cards.keys())
        random.shuffle(files)
        
        for file_path in files:
            if len(sampled) >= sample_size:
                break
            # Take 1-2 cards from each file
            file_cards = files_to_cards[file_path]
            num_from_file = min(2, len(file_cards), sample_size - len(sampled))
            sampled.extend(random.sample(file_cards, num_from_file))
        
        # Fill up with random cards if needed
        if len(sampled) < sample_size:
            remaining = [c for c in cards if c not in sampled]
            additional = min(sample_size - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, additional))
        
        # Verify we're under the size threshold, trim if needed
        sampled_size = sum(
            len(card.get("content", "")) + 
            len(card.get("peek_head", "")) + 
            len(card.get("peek_tail", ""))
            for card in sampled
        )
        
        # If still over threshold, remove cards until we fit
        while sampled_size > size_threshold and len(sampled) > 1:
            # Remove the smallest card
            smallest_idx = 0
            smallest_size = len(sampled[0].get("content", ""))
            for i, card in enumerate(sampled[1:], 1):
                card_size = len(card.get("content", ""))
                if card_size < smallest_size:
                    smallest_size = card_size
                    smallest_idx = i
            
            removed_card = sampled.pop(smallest_idx)
            sampled_size -= (len(removed_card.get("content", "")) + 
                           len(removed_card.get("peek_head", "")) + 
                           len(removed_card.get("peek_tail", "")))
        
        if self.debug:
            print(f"      Sampled {len(sampled)} cards from {len(set(c.get('relpath', 'unknown') for c in sampled))} files (final size: {sampled_size:,} chars)")
        
        return sampled
    
    def _load_manifest(self, manifest_dir: Path) -> tuple:
        """Load manifest and cards"""
        
        with open(manifest_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        cards = []
        with open(manifest_dir / "cards.jsonl") as f:
            for line in f:
                cards.append(json.loads(line))
        
        return manifest, cards
    
    def _save_results(self, output_dir: Path, manifest: Dict) -> Dict:
        """Save all graphs and analysis results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "manifest": manifest,
            "timestamp": time.time(),
            "iterations": self.iteration,
            "graphs": {},
            "total_nodes": 0,
            "total_edges": 0,
            "card_references": {}  # Map of card IDs to their content
        }
        
        # Collect all referenced cards (nodes and edges)
        all_card_ids = set()
        for graph in self.graphs.values():
            for node in graph.nodes.values():
                all_card_ids.update(node.source_refs)
            for edge in graph.edges.values():
                if hasattr(edge, 'evidence') and edge.evidence:
                    all_card_ids.update(edge.evidence)
        
        # Save each graph
        for name, graph in self.graphs.items():
            graph_data = {
                "name": graph.metadata.get("display_name", graph.name),
                "internal_name": graph.name,
                "focus": graph.focus,
                "nodes": [asdict(n) for n in graph.nodes.values()],
                "edges": [asdict(e) for e in graph.edges.values()],
                "metadata": graph.metadata,
                "stats": {
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "node_types": list(set(n.type for n in graph.nodes.values())),
                    "edge_types": list(set(e.type for e in graph.edges.values()))
                }
            }
            
            # Save individual graph
            graph_file = output_dir / f"graph_{name}.json"
            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)
            
            results["graphs"][name] = str(graph_file)
            results["total_nodes"] += len(graph.nodes)
            results["total_edges"] += len(graph.edges)
        
        # Save card store separately for retrieval during security analysis
        # Only save cards that are actually referenced by nodes
        referenced_cards = {}
        for card_id in all_card_ids:
            if card_id in self.card_store:
                referenced_cards[card_id] = self.card_store[card_id]
        
        card_refs_file = output_dir / "card_store.json"
        with open(card_refs_file, "w") as f:
            json.dump(referenced_cards, f, indent=2)
        
        results["card_store_path"] = str(card_refs_file)
        results["cards_stored"] = len(referenced_cards)
        
        # Save combined results
        results_file = output_dir / "knowledge_graphs.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self._emit("save", f"Saved {len(self.graphs)} graphs to {output_dir}")
        self._emit("save", f"Total: {results['total_nodes']} nodes, {results['total_edges']} edges")
        self._emit("save", f"Card store: {len(referenced_cards)} cards saved")
        
        # Report connectivity stats
        for name, graph in self.graphs.items():
            orphans = self._get_orphaned_nodes(graph)
            if orphans:
                pct = (len(orphans)*100//max(1,len(graph.nodes)))
                self._emit("warn", f"{name} has {len(orphans)} disconnected nodes ({pct}%)")
        
        return results
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = str(time.time())
        hash_val = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}"
