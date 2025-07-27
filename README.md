# 🎓 Shahayak 2.0: Educational AI Platform

> **Multi-agent educational AI system with role based intelligence**

## 🤖 **Agents Overview (6 Total)**

### **🧠 Root Agent**
- **`request_processor_agent/`** - **Main entry point** that routes requests to specialized agents

### **📚 Educational Agents**
- **`answer_orchastrator_agent/`** - Academic Q&A with NCERT content retrieval
- **`kg_quiz_generating_agent_new/`** - Knowledge graph-powered quiz generation (1,833+ concepts)
- **`lesson_planning_agent/`** - Structured lesson creation with curriculum alignment
- **`syllabus_planning_agent/`** - Calendar-based academic planning

### **🎨 Content Generation Agents**
- **`diagram_generating_agent/`** - Educational diagram creation with validation
- **`image_generating_agent/`** - Contextual educational image generation

### **🔧 Supporting Components**
- **`common_agents/`** - Shared RAG and role formatting agents
- **`knowledge_graph/`** - KG with 1,833 CBSE Class 10 science concepts
- **`upload_textbook_to_index/`** - Dynamic textbook processing pipeline

## 🎭 **Role-Based Intelligence**
Each agent adapts responses based on user role:
- **👨‍🎓 Student**: Motivational, 4-5 questions
- **👨‍👩‍👧‍👦 Parent**: Simple language, 3-4 questions  
- **👩‍🏫 Teacher**: Professional, 12-15 comprehensive questions

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install dependencies
pip install -r requirements.txt

# Google Cloud authentication
gcloud auth application-default login
```

### **Run locally**
```bash
# Deploy all agents to Vertex AI and start web interface
adk web
```

### **Run the System**
```bash
# Deploy all agents to Vertex AI and start web interface
python main.py
```

This will:
1. Deploy `request_processor_agent` as the **root agent**
2. Deploy all 6 specialized agents
3. Start the ADK web interface for testing
4. Show deployment URLs and status

## ⚙️ **Configuration**

Set your Google Cloud project in `main.py`:
```python
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"
```

---

**Entry Point**: `request_processor_agent` routes all requests to appropriate specialized agents.

*You can also hit the individual agents