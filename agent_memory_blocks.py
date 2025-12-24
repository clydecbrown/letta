"""
Visualize Letta Agents and Memory Blocks as an interactive graph.
Fetch all agents and blocks from the Letta API and creates
an interactive HTML visualization showing their relationships.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from ccb_essentials.filesystem import assert_real_dir, assert_real_file
from letta_client import Letta
from pyvis.network import Network

log = logging.getLogger(__name__)


def get_api_key_from_settings() -> str:
    """
    Extract LETTA_API_KEY from ~/.letta/settings.json.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If the API key cannot be retrieved.
    """
    try:
        # Use jq to extract the API key from settings.json
        settings_path = assert_real_file("~/.letta/settings.json")
        log.debug("read LETTA_API_KEY from %s", settings_path)
        result = subprocess.run(
            ["jq", "-r", ".env.LETTA_API_KEY", settings_path],
            capture_output=True,
            text=True,
            check=True
        )
        api_key = result.stdout.strip()

        if not api_key or api_key == "null":
            raise RuntimeError(
                f"LETTA_API_KEY not found in {settings_path}. "
                "Please set it in your Letta settings or use the LETTA_API_KEY environment variable."
            )

        return api_key

    except FileNotFoundError as e:
        if "jq" in str(e):
            raise RuntimeError(
                "jq command not found. Please install jq or set the LETTA_API_KEY environment variable."
            ) from e
        raise RuntimeError(
            f"Settings file not found at {settings_path}. "
            "Please ensure Letta is configured or set the LETTA_API_KEY environment variable."
        ) from e

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to read settings file: {e.stderr}. "
            "Please ensure the file is valid JSON."
        ) from e


def get_letta_client() -> Letta:
    """
    Initialize and return a Letta API client.

    Returns:
        Authenticated Letta client instance.
    """
    # Try environment variable first, then fall back to settings.json
    log.debug("Initializing Letta API client...")
    api_key = os.getenv("LETTA_API_KEY")

    if not api_key:
        log.debug("LETTA_API_KEY not found in environment")
        api_key = get_api_key_from_settings()

    try:
        client = Letta(api_key=api_key)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Letta client: {e}") from e


def save_to_cache(data: dict, cache_dir: Path, filename: str) -> None:
    """Save data to cache file as JSON."""
    cache_file = cache_dir / filename

    # Convert objects to dictionaries for JSON serialization
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

    log.debug("Cached to %s", cache_file)


def load_from_cache(cache_dir: Path, filename: str) -> dict:
    """Load data from cache file."""
    cache_file = cache_dir / filename

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    with open(cache_file, 'r') as f:
        return json.load(f)


def fetch_agents_and_blocks(
    cache_dir: Path,
    client: Letta = None,
    use_cache: bool = False,
) -> tuple[List, List, Dict[str, List[str]]]:
    """
    Fetch all agents and blocks from the Letta API or cache.

    Args:
        cache_dir: Directory to store/load cache files.
        client: Authenticated Letta client (required if not using cache).
        use_cache: If True, load from cache instead of API.

    Returns:
        Tuple of (agents_list, blocks_list, agent_to_blocks_map)
        where agent_to_blocks_map is {agent_id: [block_id1, block_id2, ...]}
    """
    if use_cache:
        log.info("Loading data from cache...")
        try:
            agents_data = load_from_cache(cache_dir, "agents.json")
            blocks_data = load_from_cache(cache_dir, "blocks.json")
            agent_to_blocks = load_from_cache(cache_dir, "agent_to_blocks.json")

            log.info("  Loaded %d agents", len(agents_data))
            log.info("  Loaded %d blocks", len(blocks_data))
            log.info("  Loaded relationships for %d agents", len(agent_to_blocks))

            # Return as-is since we're just using the data for visualization
            # The data is already in dict format which works for our purposes
            return agents_data, blocks_data, agent_to_blocks
        except FileNotFoundError as e:
            log.debug(f"  Cache miss: {e}")
            log.debug("  Falling back to API...")

    if client is None:
        raise ValueError("Client is required when not using cache")

    log.info("Fetching agents...")
    agents_response = client.agents.list()
    # Convert paginated response to list
    agents = list(agents_response)
    log.info(f"  Found {len(agents)} agents")

    log.info("Fetching all blocks...")
    blocks_response = client.blocks.list()
    # Convert paginated response to list
    all_blocks = list(blocks_response)
    log.info(f"  Found {len(all_blocks)} blocks")

    # Build map of agent -> blocks by querying each agent's blocks
    log.info("Building agent-block relationships...")
    agent_to_blocks = {}
    for agent in agents:
        try:
            agent_blocks_response = client.agents.blocks.list(agent_id=agent.id)
            # Convert paginated response to list
            agent_blocks = list(agent_blocks_response)
            agent_to_blocks[agent.id] = [block.id for block in agent_blocks]
        except Exception as e:
            log.warning(f"Failed to fetch blocks for agent {agent.id}: {e}")
            agent_to_blocks[agent.id] = []

    # Save to cache
    log.debug("\nSaving to cache...")
    save_to_cache([agent.__dict__ for agent in agents], cache_dir, "agents.json")
    save_to_cache([block.__dict__ for block in all_blocks], cache_dir, "blocks.json")
    save_to_cache(agent_to_blocks, cache_dir, "agent_to_blocks.json")

    return agents, all_blocks, agent_to_blocks


def safe_getattr(obj, attr: str, default=None):
    """Safely get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def create_visualization(
    agents: List,
    blocks: List,
    agent_to_blocks: Dict[str, List[str]],
    output_file: str = "agent_memory_blocks.html"
) -> None:
    """
    Create an interactive network visualization of agents and blocks.

    Args:
        agents: List of agent objects from Letta API.
        blocks: List of block objects from Letta API.
        agent_to_blocks: Map of agent IDs to their block IDs.
        output_file: Path to the output HTML file.
    """
    log.info(f"Creating visualization...")

    # Initialize the network with custom settings
    net = Network(
        height="100%",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False,
        directed=False
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.8
            },
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 25,
                "fit": true
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        },
        "nodes": {
            "font": {
                "size": 24,
                "face": "arial"
            },
            "borderWidth": 2
        },
        "edges": {
            "color": {
                "color": "#848484",
                "highlight": "#FF6B6B"
            },
            "width": 2,
            "smooth": {
                "type": "continuous"
            }
        }
    }
    """)

    # Add agent nodes
    log.info(f"  Adding {len(agents)} agent nodes...")
    for agent in agents:
        agent_id = safe_getattr(agent, 'id')
        agent_name = safe_getattr(agent, 'name', f'Agent {agent_id[:12]}')

        # Build tooltip with metadata
        tooltip = "Agent\n"
        tooltip += f"ID: {agent_id}\n"
        tooltip += f"Name: {agent_name}\n"

        # Add additional metadata if available
        model = safe_getattr(agent, 'model')
        if model:
            tooltip += f"Model: {model}\n"
        created_at = safe_getattr(agent, 'created_at')
        if created_at:
            tooltip += f"Created: {created_at}\n"

        net.add_node(
            agent_id,
            label=agent_name,
            title=tooltip,
            color="#87CEEB",  # Sky blue for agents
            shape="box",
            borderWidth=3,
            shapeProperties={"borderRadius": 10},  # Rounded corners for agents
            url=f"https://app.letta.com/agents/{agent_id}"
        )

    # Track which blocks are actually connected to agents
    connected_block_ids = set()
    for block_ids in agent_to_blocks.values():
        connected_block_ids.update(block_ids)

    # Add block nodes
    log.info(f"  Adding {len(blocks)} block nodes...")
    added_block_ids = set()  # Track which blocks we've added to the graph

    for block in blocks:
        block_id = safe_getattr(block, 'id')
        block_label = safe_getattr(block, 'label', f'Block {block_id[:12]}')
        block_value = safe_getattr(block, 'value', '')

        # Build tooltip with metadata
        tooltip = "Block\n"
        tooltip += f"ID: {block_id}\n"
        tooltip += f"Label: {block_label}\n"

        if block_value:
            truncated_value = truncate_text(block_value, 200)
            tooltip += f"Value: {truncated_value}\n"

        description = safe_getattr(block, 'description')
        if description:
            tooltip += f"Description: {description}\n"

        # Use different color for orphaned blocks
        is_orphaned = block_id not in connected_block_ids
        color = "#FFB6C1" if is_orphaned else "#90EE90"  # Light pink for orphaned, light green for connected

        net.add_node(
            block_id,
            label=block_label,
            title=tooltip,
            color=color,
            shape="box",
            borderWidth=2,
            shapeProperties={"borderRadius": 0}  # Square corners for blocks
        )
        added_block_ids.add(block_id)

    # Add edges between agents and blocks
    log.info(f"  Adding edges...")
    edge_count = 0
    for agent_id, block_ids in agent_to_blocks.items():
        for block_id in block_ids:
            # Check if block node exists, if not add it as a "missing" block
            if block_id not in added_block_ids:
                # This block is referenced by an agent but not in the global blocks list
                # Add it as a special "missing" node
                net.add_node(
                    block_id,
                    label=f"Block {block_id[:12]}",
                    title=f"Block\nID: {block_id}\n(Not in global blocks list)",
                    color="#FFA500",  # Orange for missing blocks
                    shape="box",
                    borderWidth=2,
                    shapeProperties={"borderRadius": 0}
                )
                added_block_ids.add(block_id)

            net.add_edge(agent_id, block_id)
            edge_count += 1

    log.info(f"    Created {edge_count} edges")

    # Add custom JavaScript for click-to-copy functionality and physics control
    custom_js = """
    <script type="text/javascript">
        // Disable physics after stabilization to stop all movement
        network.on("stabilizationIterationsDone", function() {
            network.setOptions({ physics: false });
        });
        
        // Open node URL on click
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const nodeData = nodes.get(nodeId);
                if (nodeData && nodeData.url) {
                    window.open(nodeData.url, '_blank');
                }
            }
        });
    </script>
    """

    # Save the network
    net.save_graph(output_file)

    # Inject custom CSS and JavaScript into the generated HTML
    with open(output_file, 'r') as f:
        html_content = f.read()

    # Add CSS to make the card and parent elements use full height
    custom_css = """
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        .card {
            height: 100% !important;
        }
        #mynetwork {
            height: 100% !important;
        }
    </style>
    """

    # Insert custom CSS in the head
    html_content = html_content.replace('</head>', f'{custom_css}</head>')

    # Insert custom JS before closing body tag
    html_content = html_content.replace('</body>', f'{custom_js}</body>')

    with open(output_file, 'w') as f:
        f.write(html_content)

    log.debug("Visualization saved to %s", output_file)


def print_summary(
    agents: List,
    blocks: List,
    agent_to_blocks: Dict[str, List[str]]
) -> None:
    """Print summary statistics about the graph."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Agents: {len(agents)}")
    print(f"Total Blocks: {len(blocks)}")

    # Calculate shared blocks (blocks attached to multiple agents)
    block_usage_count = {}
    for block_ids in agent_to_blocks.values():
        for block_id in block_ids:
            block_usage_count[block_id] = block_usage_count.get(block_id, 0) + 1

    shared_blocks = [bid for bid, count in block_usage_count.items() if count > 1]
    print(f"Shared Blocks (attached to multiple agents): {len(shared_blocks)}")

    # Calculate orphaned blocks (blocks not attached to any agent)
    all_block_ids = {safe_getattr(block, 'id') for block in blocks}
    connected_block_ids = set(block_usage_count.keys())
    orphaned_blocks = all_block_ids - connected_block_ids
    print(f"Orphaned Blocks (not attached to any agent): {len(orphaned_blocks)}")

    # Calculate agents with no blocks
    agents_without_blocks = sum(1 for block_ids in agent_to_blocks.values() if len(block_ids) == 0)
    print(f"Agents without blocks: {agents_without_blocks}")

    print("=" * 60)


def cleanup_orphaned_blocks(
    client: Letta,
    blocks: List,
    agent_to_blocks: Dict[str, List[str]]
) -> None:
    """Identify and delete orphaned memory blocks."""
    if client is None:
        log.error("API client required for cleanup. Do not use --cached without valid API key.")
        return

    # Identify orphaned blocks
    connected_block_ids = set()
    for b_ids in agent_to_blocks.values():
        connected_block_ids.update(b_ids)

    orphans = []
    for block in blocks:
        block_id = safe_getattr(block, 'id')
        if block_id not in connected_block_ids:
            orphans.append(block)

    if not orphans:
        print("\nNo orphaned blocks found. Cleanup not needed.")
        return

    print(f"\nFound {len(orphans)} orphaned blocks:")
    print("-" * 60)
    for i, block in enumerate(orphans, 1):
        block_id = safe_getattr(block, 'id')
        label = safe_getattr(block, 'label', 'N/A')
        value = safe_getattr(block, 'value', '')
        truncated_value = truncate_text(value, 200) if value else "Empty"
        print(f"{i}. [{block_id[:12]}] Label: {label} | Value: {truncated_value}")
        print()
        print("-" * 60)

    confirm = input(f"\nAre you sure you want to delete these {len(orphans)} blocks? (y/N): ").lower()
    if confirm != 'y':
        print("Cleanup cancelled.")
        return

    print(f"\nDeleting {len(orphans)} blocks...")
    deleted_count = 0
    for block in orphans:
        block_id = safe_getattr(block, 'id')
        try:
            client.blocks.delete(block_id=block_id)
            print(f"  ✓ Deleted {block_id[:12]}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {block_id[:12]}: {e}")

    print(f"\nCleanup complete. Successfully deleted {deleted_count} blocks.")


def main():
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize Letta Agents and Memory Blocks as an interactive graph."
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use cached API responses instead of fetching from API"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory to store/load cache files (default: cache)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Identify and delete orphaned memory blocks"
    )

    args = parser.parse_args()

    # Force live fetching if cleanup is requested
    if args.cleanup:
        args.cached = False

    cache_dir = assert_real_dir(args.cache_dir, mkdir=True)

    # Initialize Letta client only if not using cache OR if cleanup is requested
    client = None
    if not args.cached or args.cleanup:
        client = get_letta_client()

    # Fetch data (from API or cache)
    agents, blocks, agent_to_blocks = fetch_agents_and_blocks(
        cache_dir=args.cache_dir,
        client=client,
        use_cache=args.cached,
    )

    # Create visualization (save HTML to cache directory)
    output_file = cache_dir / "agent_memory_blocks.html"
    create_visualization(agents, blocks, agent_to_blocks, str(output_file))

    # Move lib directory into cache if it exists
    lib_dir = Path("lib")  # pyvis leaves this artifact in the current directory
    cache_lib_dir = cache_dir / "lib"
    if lib_dir.exists() and lib_dir.is_dir():
        # Remove existing cache lib dir if it exists
        if cache_lib_dir.exists():
            shutil.rmtree(cache_lib_dir)
        # Move lib to cache
        shutil.move(str(lib_dir), str(cache_lib_dir))
        log.debug("Moved lib directory to %s", cache_lib_dir)

    print_summary(agents, blocks, agent_to_blocks)
    print("\nOpen the HTML file in your browser to interact with the graph.")
    print("  - Hover over nodes to see metadata")
    print("  - Click on nodes to open the relevant link on Letta ADE")
    print("  - Drag nodes to rearrange the layout")
    print(f"  - {output_file}")

    # Handle cleanup if requested
    if args.cleanup:
        cleanup_orphaned_blocks(client, blocks, agent_to_blocks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
