#!/usr/bin/env python3
"""
Improved knowledge graph builder script that focuses on actual educational concepts
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from knowledge_graph.educational_kg_builder_improved import EducationalKnowledgeGraph

def main():
    """Build improved knowledge graph from your JSONL data"""
    
    print("🔬 Building IMPROVED Knowledge Graph focused on educational concepts...")
    print("   This version filters out noise and extracts meaningful scientific terms.")
    
    # Initialize the improved knowledge graph builder
    kg_builder = EducationalKnowledgeGraph()
    
    # Path to your JSONL file
    jsonl_file = "../upload_textbook_to_index/class10_science.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        return
    
    # Build the knowledge graph
    kg_builder.build_from_jsonl(jsonl_file)
    
    # Save the graph
    kg_builder.save_graph("cbse_class10_science_kg_improved")
    
    print("\n🎉 Improved knowledge graph built successfully!")
    print("📂 Files saved in: knowledge_graph/")
    print("\n🆚 Key improvements:")
    print("  ✅ Focuses on scientific vocabulary (acids, bases, photosynthesis, etc.)")
    print("  ✅ Extracts chemical formulas (H2O, CO2, NaCl, etc.)")
    print("  ✅ Filters out common words and noise")
    print("  ✅ Better definition extraction")
    print("  ✅ Meaningful relationships only")
    
    print("\nNext steps:")
    print("1. Run test_kg_improved.py to test the improved knowledge graph")
    print("2. Compare with the old results to see the difference")

if __name__ == "__main__":
    main() 