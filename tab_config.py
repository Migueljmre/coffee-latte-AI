import gradio as gr

def tab_config():
    # ================================
    # Tab 5: About & MCP Configuration
    # ================================
    with gr.TabItem("ℹ️ About & MCP Setup"):
        gr.Markdown("# 📚 Doc-MCP: Documentation RAG System")
        gr.Markdown(
            "**Transform GitHub documentation repositories into accessible MCP servers for AI agents.**"
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Project Overview
                with gr.Accordion("🎯 What is Doc-MCP?", open=True):
                    gr.Markdown("""
                    **Doc-MCP** converts GitHub documentation into AI-queryable knowledge bases via the Model Context Protocol.
                    
                    **🔑 Key Features:**
                    - 📥 **GitHub Integration** - Automatic markdown file extraction
                    - 🧠 **AI Embeddings** - Nebius AI-powered vector search  
                    - 🔍 **Smart Search** - Semantic, keyword & hybrid modes
                    - 🤖 **MCP Server** - Direct AI agent integration
                    - ⚡ **Real-time** - Live processing progress
                    """)

                # Quick Start Guide
                with gr.Accordion("🚀 Quick Start", open=False):
                    gr.Markdown("""
                    **1. Ingest Documentation** → Enter GitHub repo URL → Select files → Run 2-step pipeline
                    
                    **2. Query with AI** → Select repository → Ask questions → Get answers with sources
                    
                    **3. Manage Repos** → View stats → Delete old repositories
                    
                    **4. Use MCP Tools** → Configure your AI agent → Query docs directly from IDE
                    """)

            with gr.Column(scale=2):
                # MCP Server Configuration
                with gr.Accordion("🔧 MCP Server Setup", open=True):
                    gr.Markdown("### 🌐 Server URL")

                    # Server URL
                    gr.Textbox(
                        value="https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse",
                        label="MCP Endpoint",
                        interactive=False,
                        info="Copy this URL for your MCP client configuration",
                    )

                    gr.Markdown("### ⚙️ Configuration")

                    # SSE Configuration
                    with gr.Accordion("For Cursor, Windsurf, Cline", open=False):
                        sse_config = """{
    "mcpServers": {
    "doc-mcp": {
    "url": "https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse"
    }
    }
    }"""
                        gr.Code(
                            value=sse_config,
                            label="SSE Configuration",
                            language="json",
                            interactive=False,
                        )

                    # STDIO Configuration
                    with gr.Accordion(
                        "For STDIO Clients (Experimental)", open=False
                    ):
                        stdio_config = """{
    "mcpServers": {
    "doc-mcp": {
    "command": "npx",
    "args": ["mcp-remote", "https://agents-mcp-hackathon-doc-mcp.hf.space/gradio_api/mcp/sse", "--transport", "sse-only"]
    }
    }
    }"""
                        gr.Code(
                            value=stdio_config,
                            label="STDIO Configuration",
                            language="json",
                            interactive=False,
                        )

        # MCP Tools Overview
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🛠️ Available MCP Tools")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🔍 Documentation Query Tools**")
                        gr.Markdown(
                            "• `get_available_docs_repo` - List repositories"
                        )
                        gr.Markdown("• `make_query` - Search documentation with AI")

                    with gr.Column():
                        gr.Markdown("**📁 GitHub File Tools**")
                        gr.Markdown("• `list_repository_files` - Scan repo files")
                        gr.Markdown("• `get_single_file` - Fetch one file")
                        gr.Markdown("• `get_multiple_files` - Fetch multiple files")

        # Technology Stack & Project Info
        with gr.Row():
            with gr.Column():
                with gr.Accordion("⚙️ Technology Stack", open=False):
                    gr.Markdown("**🖥️ Frontend & API**")
                    gr.Markdown("• **Gradio** - Web interface & API framework")
                    gr.Markdown("• **Hugging Face Spaces** - Cloud hosting")

                    gr.Markdown("**🤖 AI & ML**")
                    gr.Markdown("• **Nebius AI** - LLM & embedding models")
                    gr.Markdown("• **LlamaIndex** - RAG framework")

                    gr.Markdown("**💾 Database & Storage**")
                    gr.Markdown("• **MongoDB Atlas** - Vector database")
                    gr.Markdown("• **GitHub API** - Source file access")

                    gr.Markdown("**🔌 Integration**")
                    gr.Markdown("• **Model Context Protocol** - AI agent standard")
                    gr.Markdown(
                        "• **Server-Sent Events** - Real-time communication"
                    )

            with gr.Column():
                with gr.Accordion("👥 Project Information", open=False):
                    gr.Markdown("**🏆 MCP Hackathon Project**")
                    gr.Markdown(
                        "Created to showcase AI agent integration with documentation systems."
                    )

                    gr.Markdown("**💡 Inspiration**")
                    gr.Markdown("• Making Gradio docs easily searchable")
                    gr.Markdown("• Leveraging Hugging Face AI ecosystem")
                    gr.Markdown(
                        "• Improving developer experience with AI assistants"
                    )

                    gr.Markdown("**🔮 Future Plans**")
                    gr.Markdown("• Support for PDF, HTML files")
                    gr.Markdown("• Multi-language documentation")
                    gr.Markdown("• Custom embedding fine-tuning")

                    gr.Markdown("**📄 License:** MIT - Free to use and modify")

        # Usage Examples
        with gr.Row():
            with gr.Column():
                with gr.Accordion("💡 Usage Examples", open=False):
                    gr.Markdown("### Example Workflow")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**📥 Step 1: Ingest Docs**")
                            gr.Code(
                                value="1. Enter: gradio-app/gradio\n2. Select markdown files\n3. Run ingestion pipeline",
                                label="Ingestion Process",
                                interactive=False,
                            )

                        with gr.Column():
                            gr.Markdown("**🤖 Step 2: Query with AI**")
                            gr.Code(
                                value='Query: "How to create custom components?"\nResponse: Detailed answer with source links',
                                label="AI Query Example",
                                interactive=False,
                            )

                    gr.Markdown("### MCP Tool Usage")
                    gr.Code(
                        value="""# In your AI agent:
    1. Call: get_available_docs_repo() -> ["gradio-app/gradio", ...]  
    2. Call: make_query("gradio-app/gradio", "default", "custom components")
    3. Get: AI response + source citations""",
                        label="MCP Integration Example",
                        language="python",
                        interactive=False,
                    )