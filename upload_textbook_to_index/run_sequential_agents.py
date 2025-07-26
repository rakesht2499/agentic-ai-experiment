#!/usr/bin/env python3
"""
Sequential Google ADK Agent Runner

This script runs the three textbook processing agents in sequence:
1. Agent 1: Parse PDFs with Document AI and save to GCS
2. Agent 2: Convert DocAI results into RAG-formatted JSON chunks
3. Agent 3: Upload processed JSON to Vertex RAG

The script handles errors gracefully and provides clear logging.
"""

import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'sequential_agents_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class SequentialAgentRunner:
    """Manages the sequential execution of the three agents"""
    
    def __init__(self, script_directory=None):
        self.script_directory = script_directory or Path(__file__).parent
        self.agents = [
            {
                'name': 'Agent 1 - PDF Processing with Document AI',
                'script': 'agent_1_parses_pdf_with_document_ai.py',
                'description': 'Processes PDFs using Google Document AI and outputs JSON to GCS'
            },
            {
                'name': 'Agent 2 - Convert to RAG Format',
                'script': 'agent_2_convert_res_into_chapter_wise_json.py',
                'description': 'Converts DocAI JSON into RAG-formatted chunks and uploads to GCS'
            },
            {
                'name': 'Agent 3 - Upload to Vertex RAG',
                'script': 'agent_3_upload_json_to_vertex_rag.py',
                'description': 'Ingests processed files from GCS into Vertex RAG'
            }
        ]
    
    def check_prerequisites(self):
        """Check if all required scripts exist and environment is ready"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if all agent scripts exist
        missing_scripts = []
        for agent in self.agents:
            script_path = self.script_directory / agent['script']
            if not script_path.exists():
                missing_scripts.append(agent['script'])
        
        if missing_scripts:
            logger.error(f"❌ Missing required scripts: {', '.join(missing_scripts)}")
            return False
        
        # Check if Python environment has required packages
        try:
            import google.cloud.documentai_v1
            import google.cloud.storage
            import vertexai
            logger.info("✅ Required Google Cloud packages are available")
        except ImportError as e:
            logger.error(f"❌ Missing required package: {e}")
            logger.info("💡 Please install required packages: pip install google-cloud-documentai google-cloud-storage google-cloud-aiplatform")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def run_agent(self, agent_info, agent_number):
        """Run a single agent script"""
        script_name = agent_info['script']
        script_path = self.script_directory / script_name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting {agent_info['name']}")
        logger.info(f"📄 Script: {script_name}")
        logger.info(f"📝 Description: {agent_info['description']}")
        logger.info(f"{'='*60}")
        
        try:
            # Run the agent script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.script_directory,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Log the output
            if result.stdout:
                logger.info(f"📤 Agent {agent_number} Output:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            
            if result.stderr:
                logger.warning(f"⚠️ Agent {agent_number} Warnings/Errors:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.warning(f"   {line}")
            
            # Check return code
            if result.returncode == 0:
                logger.info(f"✅ {agent_info['name']} completed successfully")
                return True
            else:
                logger.error(f"❌ {agent_info['name']} failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {agent_info['name']} timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"💥 {agent_info['name']} failed with exception: {e}")
            return False

    def run_all_agents(self):
        """Run all agents in sequence"""
        logger.info("🎬 Starting Sequential Agent Execution")
        logger.info(f"📂 Working directory: {self.script_directory}")
        
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Aborting execution.")
            return False
        
        start_time = datetime.now()
        successful_agents = 0
        
        for i, agent in enumerate(self.agents, 1):
            if self.run_agent(agent, i):
                successful_agents += 1
                logger.info(f"✅ Agent {i} completed successfully")
            else:
                logger.error(f"❌ Agent {i} failed. Stopping execution.")
                break
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 EXECUTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info(f"✅ Successful agents: {successful_agents}/{len(self.agents)}")
        
        if successful_agents == len(self.agents):
            logger.info("🎉 All agents completed successfully!")
            logger.info("📚 Your textbook should now be available in Vertex RAG")
            return True
        else:
            logger.error(f"⚠️  Only {successful_agents} out of {len(self.agents)} agents completed successfully")
            return False

def main():
    """Main entry point"""
    print("🤖 Google ADK Sequential Agent Runner")
    print("=====================================")
    
    runner = SequentialAgentRunner()
    
    try:
        success = runner.run_all_agents()
        if success:
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check the logs for details.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
