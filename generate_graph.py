
from pathlib import Path
from dental_agent.workflow.graph import dental_graph

# Get the graph structure
graph = dental_graph.get_graph()

# Print the graph in Mermaid format
mermaid_code = graph.draw_mermaid()
print(mermaid_code)

# Save the graph definition
Path("dental_architecture.mmd").write_text(
    mermaid_code,
    encoding="utf-8"
)

# Export as PNG
png_bytes = graph.draw_mermaid_png()

Path("dental_architecture.png").write_bytes(png_bytes)

print("Saved: dental_architecture.png")
