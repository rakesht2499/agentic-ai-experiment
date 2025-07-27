#!/usr/bin/env python3
"""
Fixed knowledge graph builder script
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from knowledge_graph.educational_kg_builder_fixed import EducationalKnowledgeGraph

def main():
    """Build knowledge graph from your JSONL data"""
    
    # Initialize the knowledge graph builder
    kg_builder = EducationalKnowledgeGraph()
    
    # Path to your JSONL file
    jsonl_file = "../upload_textbook_to_index/class10_science.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        return
    
    print("🚀 Building knowledge graph from textbook content...")
    
    # Build the knowledge graph
    kg_builder.build_from_jsonl(jsonl_file)
    
    # Save the graph
    kg_builder.save_graph("cbse_class10_science_kg")
    
    print("🎉 Knowledge graph built successfully!")
    print("📂 Files saved in: knowledge_graph/")
    print("\nNext steps:")
    print("1. Run test_kg_queries.py to test the knowledge graph")
    print("2. Integrate with your quiz and answer agents")

if __name__ == "__main__":
    main()