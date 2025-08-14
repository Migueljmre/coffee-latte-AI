import asyncio
import os
import time
import traceback
from typing import Dict, List

import gradio as gr
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.text_splitter import SentenceSplitter

from rag.config import (
    delete_repository_data,
    embed_model,
    get_available_repos,
    get_repo_details,
    get_repository_stats,
    llm,
)
from rag.github_file_loader import fetch_markdown_files as fetch_files_with_loader
from rag.github_file_loader import fetch_repository_files, load_github_files
from rag.ingest import ingest_documents_async
from rag.query import QueryRetriever

load_dotenv()

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=3072)

# Environment variable to control repository management visibility
ENABLE_REPO_MANAGEMENT = os.getenv("ENABLE_REPO_MANAGEMENT", "true").lower() == "true"

def get_available_repositories():
    return get_available_repos()


def start_file_loading(
    repo_url: str, selected_files: List[str], current_progress: Dict
):
    """Step 1: Load files from GitHub"""
    print("\n🔄 STARTING FILE LOADING STEP")
    print(f"📍 Repository: {repo_url}")
    print(f"📋 Selected files: {selected_files}")

    if not selected_files:
        return {
            "status": "error",
            "message": "❌ No files selected for loading",
            "progress": 0,
            "details": "",
            "step": "file_loading",
        }

    total_files = len(selected_files)
    start_time = time.time()

    # Parse repo name from URL
    if "github.com" in repo_url:
        repo_name = (
            repo_url.replace("https://github.com/", "")
            .replace("http://github.com/", "")
            .strip("/")
        )
        if "/" not in repo_name:
            return {
                "status": "error",
                "message": "❌ Invalid repository URL format",
                "progress": 0,
                "details": "",
                "step": "file_loading",
            }
    else:
        repo_name = repo_url.strip()

    try:
        batch_size = 25
        all_documents = []
        all_failed = []

        current_progress.update(
            {
                "status": "loading",
                "message": f"🚀 Loading files from {repo_name}",
                "progress": 0,
                "total_files": total_files,
                "processed_files": 0,
                "phase": "File Loading",
                "details": f"Processing {total_files} files in batches...",
                "step": "file_loading",
            }
        )

        for i in range(0, len(selected_files), batch_size):
            batch = selected_files[i : i + batch_size]

            print(f"\n📦 PROCESSING BATCH {i // batch_size + 1}")
            print(f"   Files: {batch}")

            # Update progress for current batch
            progress_percentage = (i / total_files) * 100
            current_progress.update(
                {
                    "progress": progress_percentage,
                    "processed_files": i,
                    "current_batch": i // batch_size + 1,
                    "details": f"Loading batch {i // batch_size + 1}: {', '.join([f.split('/')[-1] for f in batch])}",
                }
            )

            try:
                documents, failed = load_github_files(
                    repo_name=repo_name,
                    file_paths=batch,
                    branch="main",
                    concurrent_requests=10,
                    github_token=os.getenv("GITHUB_API_KEY"),
                )

                print("✅ Load results:")
                print(f"   - Documents: {len(documents)}")
                print(f"   - Failed: {len(failed)}")

                if documents:
                    for j, doc in enumerate(documents):
                        print(f"   📄 Doc {j + 1}: {doc.doc_id}")
                        print(f"      Size: {len(doc.text)} chars")

                        # Ensure repo metadata is set
                        if "repo" not in doc.metadata:
                            doc.metadata["repo"] = repo_name
                            print(f"      ✅ Added repo metadata: {repo_name}")

                all_documents.extend(documents)
                all_failed.extend(failed)

            except Exception as batch_error:
                print(f"❌ Batch processing error: {batch_error}")
                all_failed.extend(batch)

        loading_time = time.time() - start_time

        # Store loaded documents in progress state for next step
        current_progress.update(
            {
                "status": "loaded",
                "message": f"✅ File Loading Complete! Loaded {len(all_documents)} documents",
                "progress": 100,
                "phase": "Files Loaded",
                "details": f"Successfully loaded {len(all_documents)} documents in {loading_time:.1f}s",
                "step": "file_loading_complete",
                "loaded_documents": all_documents,  # Store documents for next step
                "failed_files": all_failed,
                "loading_time": loading_time,
                "repo_name": repo_name,
            }
        )

        return current_progress

    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"❌ File loading error after {total_time:.1f}s: {str(e)}"
        print(error_msg)

        current_progress.update(
            {
                "status": "error",
                "message": error_msg,
                "progress": 0,
                "phase": "Failed",
                "details": str(e),
                "error": str(e),
                "step": "file_loading",
            }
        )

        return current_progress


def start_vector_ingestion(current_progress: Dict):
    """Step 2: Ingest loaded documents into vector store"""
    print("\n🔄 STARTING VECTOR INGESTION STEP")

    # Check if we have loaded documents from previous step
    if current_progress.get("step") != "file_loading_complete":
        return {
            "status": "error",
            "message": "❌ No loaded documents found. Please load files first.",
            "progress": 0,
            "details": "",
            "step": "vector_ingestion",
        }

    all_documents = current_progress.get("loaded_documents", [])
    repo_name = current_progress.get("repo_name", "")

    if not all_documents:
        return {
            "status": "error",
            "message": "❌ No documents available for vector ingestion",
            "progress": 0,
            "details": "",
            "step": "vector_ingestion",
        }

    vector_start_time = time.time()

    # Update state for vector store phase
    current_progress.update(
        {
            "status": "vectorizing",
            "message": "🔄 Generating embeddings and storing in vector database",
            "progress": 0,
            "phase": "Vector Store Ingestion",
            "details": f"Processing {len(all_documents)} documents for embedding...",
            "step": "vector_ingestion",
        }
    )

    try:
        print("🔄 STARTING VECTOR STORE INGESTION")
        print(f"   Repository: {repo_name}")
        print(f"   Documents to process: {len(all_documents)}")

        # Call the async ingestion function with repo name
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ingest_documents_async(all_documents, repo_name))
        finally:
            loop.close()

        vector_time = time.time() - vector_start_time
        loading_time = current_progress.get("loading_time", 0)
        total_time = loading_time + vector_time

        print(f"✅ Vector ingestion completed in {vector_time:.2f} seconds")

        failed_files_data = current_progress.get("failed_files", [])
        if isinstance(failed_files_data, list):
            failed_files_count = len(failed_files_data)
        else:
            failed_files_count = (
                failed_files_data if isinstance(failed_files_data, int) else 0
            )

        # Update final success state with repository update flag
        current_progress.update(
            {
                "status": "complete",
                "message": "✅ Complete Ingestion Pipeline Finished!",
                "progress": 100,
                "phase": "Complete",
                "details": f"Successfully processed {len(all_documents)} documents for {repo_name}",
                "step": "complete",
                "total_time": total_time,
                "documents_processed": len(all_documents),
                "failed_files_count": failed_files_count,  # Use count instead of trying len()
                "failed_files": failed_files_data,  # Keep original data
                "vector_time": vector_time,
                "loading_time": loading_time,
                "repo_name": repo_name,
                "repository_updated": True,  # Flag to trigger repo list refresh
            }
        )

        return current_progress

    except Exception as ingest_error:
        vector_time = time.time() - vector_start_time
        print(f"❌ Vector ingestion failed after {vector_time:.2f} seconds")
        print(f"❌ Error: {ingest_error}")

        # Get failed files data safely
        failed_files_data = current_progress.get("failed_files", [])
        if isinstance(failed_files_data, list):
            failed_files_count = len(failed_files_data)
        else:
            failed_files_count = (
                failed_files_data if isinstance(failed_files_data, int) else 0
            )

        current_progress.update(
            {
                "status": "error",
                "message": "❌ Vector Store Ingestion Failed",
                "progress": 0,
                "phase": "Failed",
                "details": f"Error: {str(ingest_error)}",
                "error": str(ingest_error),
                "step": "vector_ingestion",
                "failed_files_count": failed_files_count,
                "failed_files": failed_files_data,
            }
        )

        return current_progress


def start_file_loading_generator(
    repo_url: str, selected_files: List[str], current_progress: Dict
):
    """Step 1: Load files from GitHub with yield-based real-time updates"""

    print("\n🔄 STARTING FILE LOADING STEP")
    print(f"📍 Repository: {repo_url}")
    print(f"📋 Selected files: {len(selected_files)} files")

    if not selected_files:
        error_progress = {
            "status": "error",
            "message": "❌ No files selected for loading",
            "progress": 0,
            "details": "Please select at least one file to proceed.",
            "step": "file_loading",
        }
        yield error_progress
        return error_progress

    total_files = len(selected_files)
    start_time = time.time()

    # Parse repo name from URL
    if "github.com" in repo_url:
        repo_name = (
            repo_url.replace("https://github.com/", "")
            .replace("http://github.com/", "")
            .strip("/")
        )
        if "/" not in repo_name:
            error_progress = {
                "status": "error",
                "message": "❌ Invalid repository URL format",
                "progress": 0,
                "details": "Expected format: owner/repo or https://github.com/owner/repo",
                "step": "file_loading",
            }
            yield error_progress
            return error_progress
    else:
        repo_name = repo_url.strip()

    try:
        batch_size = 10
        all_documents = []
        all_failed = []

        # Initial progress update
        initial_progress = {
            "status": "loading",
            "message": f"🚀 Starting file loading from {repo_name}",
            "progress": 0,
            "total_files": total_files,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "phase": "File Loading",
            "details": f"Preparing to load {total_files} files in batches of {batch_size}...",
            "step": "file_loading",
            "current_batch": 0,
            "total_batches": (len(selected_files) + batch_size - 1) // batch_size,
            "repo_name": repo_name,
        }
        yield initial_progress

        time.sleep(0.5)

        for i in range(0, len(selected_files), batch_size):
            batch = selected_files[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (len(selected_files) + batch_size - 1) // batch_size

            # Update progress at batch start
            batch_start_progress = {
                "status": "loading",
                "message": f"🔄 Loading batch {current_batch_num}/{total_batches}",
                "progress": (i / total_files) * 90,
                "processed_files": i,
                "successful_files": len(all_documents),
                "failed_files": len(all_failed),
                "current_batch": current_batch_num,
                "total_batches": total_batches,
                "phase": "File Loading",
                "details": f"Processing batch {current_batch_num}: {', '.join([f.split('/')[-1] for f in batch[:3]])}{'...' if len(batch) > 3 else ''}",
                "step": "file_loading",
                "repo_name": repo_name,
            }
            yield batch_start_progress

            try:
                print(f"\n📦 PROCESSING BATCH {current_batch_num}/{total_batches}")
                print(f"   Files: {[f.split('/')[-1] for f in batch]}")

                documents, failed = load_github_files(
                    repo_name=repo_name,
                    file_paths=batch,
                    branch="main",
                    concurrent_requests=10,
                    github_token=os.getenv("GITHUB_API_KEY"),
                )

                print("✅ Load results:")
                print(f"   - Documents: {len(documents)}")
                print(f"   - Failed: {len(failed)}")

                # Process documents
                for j, doc in enumerate(documents):
                    print(f"   📄 Doc {j + 1}: {doc.doc_id}")
                    print(f"      Size: {len(doc.text)} chars")

                    if "repo" not in doc.metadata:
                        doc.metadata["repo"] = repo_name
                        print(f"      ✅ Added repo metadata: {repo_name}")

                all_documents.extend(documents)
                all_failed.extend(failed)

                # Update progress after batch completion
                batch_complete_progress = {
                    "status": "loading",
                    "message": f"✅ Completed batch {current_batch_num}/{total_batches}",
                    "progress": ((i + len(batch)) / total_files) * 90,
                    "processed_files": i + len(batch),
                    "successful_files": len(all_documents),
                    "failed_files": len(all_failed),
                    "current_batch": current_batch_num,
                    "total_batches": total_batches,
                    "phase": "File Loading",
                    "details": f"✅ Batch {current_batch_num} complete: {len(documents)} loaded, {len(failed)} failed. Total progress: {len(all_documents)} documents loaded.",
                    "step": "file_loading",
                    "repo_name": repo_name,
                }
                yield batch_complete_progress

                time.sleep(0.3)

            except Exception as batch_error:
                print(f"❌ Batch processing error: {batch_error}")
                all_failed.extend(batch)

                error_progress = {
                    "status": "loading",
                    "message": f"⚠️ Error in batch {current_batch_num}",
                    "progress": ((i + len(batch)) / total_files) * 90,
                    "processed_files": i + len(batch),
                    "successful_files": len(all_documents),
                    "failed_files": len(all_failed),
                    "current_batch": current_batch_num,
                    "phase": "File Loading",
                    "details": f"❌ Batch {current_batch_num} error: {str(batch_error)[:100]}... Continuing with next batch.",
                    "step": "file_loading",
                    "repo_name": repo_name,
                }
                yield error_progress

        loading_time = time.time() - start_time

        # Final completion update
        completion_progress = {
            "status": "loaded",
            "message": f"✅ File Loading Complete! Loaded {len(all_documents)} documents",
            "progress": 100,
            "phase": "Files Loaded Successfully",
            "details": f"🎯 Final Results:\n✅ Successfully loaded: {len(all_documents)} documents\n❌ Failed files: {len(all_failed)}\n⏱️ Total time: {loading_time:.1f}s\n📊 Success rate: {(len(all_documents) / (len(all_documents) + len(all_failed)) * 100):.1f}%",
            "step": "file_loading_complete",
            "loaded_documents": all_documents,
            "failed_files": all_failed,
            "loading_time": loading_time,
            "repo_name": repo_name,
            "total_files": total_files,
            "processed_files": total_files,
            "successful_files": len(all_documents),
        }
        yield completion_progress
        return completion_progress

    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"❌ File loading error after {total_time:.1f}s: {str(e)}"
        print(error_msg)

        error_progress = {
            "status": "error",
            "message": error_msg,
            "progress": 0,
            "phase": "Loading Failed",
            "details": f"Critical error during file loading:\n{str(e)}",
            "error": str(e),
            "step": "file_loading",
        }
        yield error_progress
        return error_progress


# Progress display component
def format_progress_display(progress_state: Dict) -> str:
    """Format progress state into readable display with enhanced details"""
    if not progress_state:
        return "🚀 Ready to start ingestion...\n\n📋 **Two-Step Process:**\n1️⃣ Load files from GitHub repository\n2️⃣ Generate embeddings and store in vector database"

    status = progress_state.get("status", "unknown")
    message = progress_state.get("message", "")
    progress = progress_state.get("progress", 0)
    phase = progress_state.get("phase", "")
    details = progress_state.get("details", "")

    # Enhanced progress bar
    filled = int(progress / 2.5)  # 40 chars total
    progress_bar = "█" * filled + "░" * (40 - filled)

    # Status emoji mapping
    status_emoji = {
        "loading": "⏳",
        "loaded": "✅",
        "vectorizing": "🧠",
        "complete": "🎉",
        "error": "❌",
    }

    emoji = status_emoji.get(status, "🔄")

    output = f"{emoji} **{message}**\n\n"

    # Phase and progress section
    output += f"📊 **Current Phase:** {phase}\n"
    output += f"📈 **Progress:** {progress:.1f}%\n"
    output += f"[{progress_bar}] {progress:.1f}%\n\n"

    # Step-specific details for file loading
    if progress_state.get("step") == "file_loading":
        processed = progress_state.get("processed_files", 0)
        total = progress_state.get("total_files", 0)
        successful = progress_state.get("successful_files", 0)
        failed = progress_state.get("failed_files", 0)

        if total > 0:
            output += "📁 **File Processing Status:**\n"
            output += f"   • Total files: {total}\n"
            output += f"   • Processed: {processed}/{total}\n"
            output += f"   • ✅ Successful: {successful}\n"
            output += f"   • ❌ Failed: {failed}\n"

            if "current_batch" in progress_state and "total_batches" in progress_state:
                output += f"   • 📦 Current batch: {progress_state['current_batch']}/{progress_state['total_batches']}\n"
            output += "\n"

    # Step-specific details for vector ingestion
    elif progress_state.get("step") == "vector_ingestion":
        docs_count = progress_state.get("documents_count", 0)
        repo_name = progress_state.get("repo_name", "Unknown")

        if docs_count > 0:
            output += "🧠 **Vector Processing Status:**\n"
            output += f"   • Repository: {repo_name}\n"
            output += f"   • Documents: {docs_count:,}\n"
            output += f"   • Stage: {phase}\n\n"

    # Detailed information
    output += f"📝 **Details:**\n{details}\n"

    # Final summary for completion
    if status == "complete":
        total_time = progress_state.get("total_time", 0)
        docs_processed = progress_state.get("documents_processed", 0)
        failed_files = progress_state.get("failed_files", 0)
        vector_time = progress_state.get("vector_time", 0)
        loading_time = progress_state.get("loading_time", 0)
        repo_name = progress_state.get("repo_name", "Unknown")

        output += "\n🎊 **INGESTION COMPLETED SUCCESSFULLY!**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"🎯 **Repository:** {repo_name}\n"
        output += f"📄 **Documents processed:** {docs_processed:,}\n"
        output += f"❌ **Failed files:** {len(failed_files) if isinstance(failed_files, list) else failed_files}\n"
        output += f"⏱️ **Total time:** {total_time:.1f} seconds\n"
        output += f"   ├─ File loading: {loading_time:.1f}s\n"
        output += f"   └─ Vector processing: {vector_time:.1f}s\n"
        output += (
            f"📊 **Processing rate:** {docs_processed / total_time:.1f} docs/second\n\n"
        )
        output += "🚀 **Next Step:** Go to the 'Query Interface' tab to start asking questions!"

    elif status == "error":
        error = progress_state.get("error", "Unknown error")
        output += "\n💥 **ERROR OCCURRED**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += (
            f"❌ **Error Details:** {error[:300]}{'...' if len(error) > 300 else ''}\n"
        )
        output += "\n🔧 **Troubleshooting Tips:**\n"
        output += "   • Check your GitHub token permissions\n"
        output += "   • Verify repository URL format\n"
        output += "   • Ensure selected files exist\n"
        output += "   • Check network connectivity\n"

    return output


# Create the main Gradio interface
with gr.Blocks(title="Doc-MCP") as demo:
    gr.Markdown("# 📚Doc-MCP: Documentation RAG System")
    gr.Markdown(
        "Transform GitHub documentation repositories into accessible MCP (Model Context Protocol) servers for AI agents. Upload documentation, generate vector embeddings, and query with intelligent context retrieval."
    )

    # State variables
    files_state = gr.State([])
    progress_state = gr.State({})

    with gr.Tabs():
        with gr.TabItem("📥 Documentation Ingestion"):
            gr.Markdown("### 🚀 Two-Step Documentation Processing Pipeline")
            gr.Markdown(
                "**Step 1:** Fetch markdown files from GitHub repository → **Step 2:** Generate vector embeddings and store in MongoDB Atlas"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    repo_input = gr.Textbox(
                        label="📂 GitHub Repository URL",
                        placeholder="Enter: owner/repo or https://github.com/owner/repo (e.g., gradio-app/gradio)",
                        value="",
                        info="Enter any GitHub repository containing markdown documentation",
                    )
                    load_btn = gr.Button(
                        "🔍 Discover Documentation Files", variant="secondary"
                    )

                with gr.Column(scale=1):
                    status_output = gr.Textbox(
                        label="Repository Discovery Status",
                        interactive=False,
                        lines=4,
                        placeholder="Repository scanning results will appear here...",
                    )
            with gr.Row():
                select_all_btn = gr.Button(
                    "📋 Select All Documents", variant="secondary"
                )
                clear_all_btn = gr.Button("🗑️ Clear Selection", variant="secondary")

            # File selection
            with gr.Accordion(label="Available Documentation Files"):
                file_selector = gr.CheckboxGroup(
                    choices=[],
                    label="Select Markdown Files for RAG Processing",
                    visible=False,
                )

            # Two-step ingestion controls
            gr.Markdown("### 🔄 RAG Pipeline Execution")
            gr.Markdown(
                "Process your documentation through our advanced RAG pipeline using Nebius AI embeddings and MongoDB Atlas vector storage."
            )

            with gr.Row():
                with gr.Column():
                    step1_btn = gr.Button(
                        "📥 Step 1: Load Files from GitHub",
                        variant="primary",
                        size="lg",
                        interactive=False,
                    )

                with gr.Column():
                    step2_btn = gr.Button(
                        "🔄 Step 2: Start Ingestion",
                        variant="primary",
                        size="lg",
                        interactive=False,
                    )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Progress", variant="secondary")
                reset_btn = gr.Button("🗑️ Reset Progress", variant="secondary")

            # Progress display
            progress_display = gr.Textbox(
                label="📊 Real-time Ingestion Progress",
                interactive=False,
                lines=25,
                value="🚀 Ready to start two-step ingestion process...\n\n📋 Steps:\n1️⃣ Load files from GitHub repository\n2️⃣ Generate embeddings and store in vector database",
                max_lines=30,
            )

            # Event handlers
            def load_files_handler(repo_url: str):
                if not repo_url.strip():
                    return (
                        gr.CheckboxGroup(choices=[], visible=False),
                        "Please enter a repository URL",
                        [],
                        gr.Button(interactive=False),
                        gr.Button(interactive=False),
                    )

                files, message = fetch_files_with_loader(repo_url)

                if files:
                    return (
                        gr.CheckboxGroup(
                            choices=files,
                            value=[],
                            label=f"Select Files from {repo_url} ({len(files)} files)",
                            visible=True,
                        ),
                        message,
                        files,
                        gr.Button(interactive=True),  # Enable step 1 button
                        gr.Button(interactive=False),  # Keep step 2 disabled
                    )
                else:
                    return (
                        gr.CheckboxGroup(choices=[], visible=False),
                        message,
                        [],
                        gr.Button(interactive=False),
                        gr.Button(interactive=False),
                    )

            def start_step1_generator(
                repo_url: str, selected_files: List[str], current_progress: Dict
            ):
                """Start Step 1 with generator-based real-time progress updates"""
                for progress_update in start_file_loading_generator(
                    repo_url, selected_files, current_progress.copy()
                ):
                    progress_text = format_progress_display(progress_update)
                    step2_enabled = (
                        progress_update.get("step") == "file_loading_complete"
                    )

                    yield (
                        progress_update,
                        progress_text,
                        gr.Button(interactive=step2_enabled),
                    )

            def start_step2(current_progress: Dict):
                """Start Step 2: Vector Ingestion"""
                new_progress = start_vector_ingestion(current_progress.copy())
                progress_text = format_progress_display(new_progress)
                return new_progress, progress_text

            def refresh_progress(current_progress: Dict):
                """Refresh the progress display"""
                progress_text = format_progress_display(current_progress)
                return progress_text

            def reset_progress():
                """Reset all progress"""
                return (
                    {},
                    "Ready to start two-step ingestion process...",
                    gr.Button(interactive=False),
                )

            def select_all_handler(available_files):
                if available_files:
                    return gr.CheckboxGroup(value=available_files)
                return gr.CheckboxGroup(value=[])

            def clear_all_handler():
                return gr.CheckboxGroup(value=[])

            # Wire up events
            load_btn.click(
                fn=load_files_handler,
                inputs=[repo_input],
                outputs=[
                    file_selector,
                    status_output,
                    files_state,
                    step1_btn,
                    step2_btn,
                ],
                show_api=False,
            )

            select_all_btn.click(
                fn=select_all_handler,
                inputs=[files_state],
                outputs=[file_selector],
                show_api=False,
            )

            clear_all_btn.click(
                fn=clear_all_handler, outputs=[file_selector], show_api=False
            )

            step1_btn.click(
                fn=start_step1_generator,
                inputs=[repo_input, file_selector, progress_state],
                outputs=[progress_state, progress_display, step2_btn],
                show_api=False,
            )

            step2_btn.click(
                fn=start_step2,
                inputs=[progress_state],
                outputs=[progress_state, progress_display],
                show_api=False,
            )

            refresh_btn.click(
                fn=refresh_progress,
                inputs=[progress_state],
                outputs=[progress_display],
                show_api=False,
            )

            reset_btn.click(
                fn=reset_progress,
                outputs=[progress_state, progress_display, step2_btn],
                show_api=False,
            )

        # ================================
        # Tab 2: Query Interface
        # ================================
        with gr.TabItem("🤖 AI Documentation Assistant"):
            gr.Markdown("### 💬 Intelligent Documentation Q&A")
            gr.Markdown(
                "Query your processed documentation using advanced semantic search. Get contextual answers with source citations powered by Nebius LLM and vector similarity search."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Repository selection - Dropdown that becomes textbox when selected
                    with gr.Row():
                        repo_dropdown = gr.Dropdown(
                            choices=get_available_repositories()
                            or ["No repositories available"],
                            label="📚 Select Documentation Repository",
                            value=None,
                            interactive=True,
                            allow_custom_value=True,
                            info="Choose from available repositories",
                        )

                        # Hidden textbox that will become visible when repo is selected
                        selected_repo_textbox = gr.Textbox(
                            label="🎯 Selected Repository",
                            value="",
                            interactive=False,
                            visible=False,
                            info="Currently selected repository for querying",
                        )

                    refresh_repos_btn = gr.Button(
                        "🔄 Refresh Repository List", variant="secondary", size="sm"
                    )

                    # Query mode selection
                    query_mode = gr.Radio(
                        choices=["default", "text_search", "hybrid"],
                        label="🔍 Search Strategy",
                        value="default",
                        info="• default: Semantic similarity (AI understanding)\n• text_search: Keyword matching\n• hybrid: Combined approach for best results",
                    )

                    # Query input
                    query_input = gr.Textbox(
                        label="💭 Ask About Your Documentation",
                        placeholder="How do I implement a custom component? What are the available API endpoints? How to configure the system?",
                        lines=3,
                        info="Ask natural language questions about your documentation",
                    )

                    query_btn = gr.Button(
                        "🚀 Search Documentation", variant="primary", size="lg"
                    )

                    # Response display as text area
                    response_output = gr.Textbox(
                        label="🤖 AI Assistant Response",
                        value="Your AI-powered documentation response will appear here with contextual information and source citations...",
                        lines=10,
                        interactive=False,
                        info="Generated using Nebius LLM with retrieved documentation context",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📖 Source References")
                    gr.Markdown(
                        "View the exact documentation sources used to generate the response, with relevance scores and GitHub links."
                    )

                    # Source nodes display as JSON
                    sources_output = gr.JSON(
                        label="📎 Source Citations & Metadata",
                        value={
                            "message": "Source documentation excerpts with relevance scores will appear here after your query...",
                            "info": "Each source includes file path, relevance score, and content snippet",
                        },
                    )

            # Event handlers
            def handle_repo_selection(selected_repo):
                """Handle repository selection from dropdown"""
                if not selected_repo or selected_repo in [
                    "No repositories available",
                    "",
                ]:
                    return (
                        gr.Dropdown(visible=True),  # Keep dropdown visible
                        gr.Textbox(visible=False, value=""),  # Hide textbox
                        gr.Button(interactive=False),  # Disable query button
                    )
                else:
                    return (
                        gr.Dropdown(visible=False),  # Hide dropdown
                        gr.Textbox(
                            visible=True, value=selected_repo
                        ),  # Show textbox with selected repo
                        gr.Button(interactive=True),  # Enable query button
                    )

            def reset_repo_selection():
                """Reset to show dropdown again"""
                try:
                    repos = get_available_repositories() or [
                        "No repositories available"
                    ]
                    return (
                        gr.Dropdown(
                            choices=repos, value=None, visible=True
                        ),  # Show dropdown with refreshed choices
                        gr.Textbox(visible=False, value=""),  # Hide textbox
                        gr.Button(interactive=False),  # Disable query button
                    )
                except Exception as e:
                    print(f"Error refreshing repository list: {e}")
                    return (
                        gr.Dropdown(
                            choices=["Error loading repositories"],
                            value=None,
                            visible=True,
                        ),
                        gr.Textbox(visible=False, value=""),
                        gr.Button(interactive=False),
                    )

            def get_available_docs_repo():
                """
                List the available docs of repositories - should be called first to list out all the available repo docs to chat with

                Returns:
                    Updated dropdown with available repositories
                """
                try:
                    repos = get_available_repositories()
                    if not repos:
                        repos = [
                            "No repositories available - Please ingest documentation first"
                        ]
                    return gr.Dropdown(choices=repos, value=None)
                except Exception as e:
                    print(f"Error refreshing repository list: {e}")
                    return gr.Dropdown(
                        choices=["Error loading repositories"], value=None
                    )

            # Simple query handler
            def handle_query(repo: str, mode: str, query: str):
                """
                Handle query request - returns raw data from retriever
                Args:
                    repo: Selected repository from textbox
                    mode: Query mode (default, text_search, hybrid)
                    query: User's query
                Returns:
                    Raw result dict from QueryRetriever.make_query()
                """
                if not query.strip():
                    return {"error": "Please enter a query."}

                if not repo or repo in [
                    "No repositories available",
                    "Error loading repositories",
                    "",
                ]:
                    return {"error": "Please select a valid repository."}

                try:
                    # Create query retriever for the selected repo
                    retriever = QueryRetriever(repo)

                    # Make the query and return raw result
                    result = retriever.make_query(query, mode)
                    return result

                except Exception as e:
                    print(f"Query error: {e}")
                    traceback.print_exc()
                    return {"error": f"Query failed: {str(e)}"}

            def make_query(repo: str, mode: str, query: str):
                """
                Retrieve relevant documentation context for a given query using specified retrieval mode.

                This function is designed to support Retrieval-Augmented Generation (RAG) by extracting
                the most relevant context chunks from indexed documentation sources.
                Args:
                    repo: Selected repository from the textbox input
                    mode: Query mode (default, text_search, hybrid)
                    query: User's query
                Returns:
                    Tuple of (response_text, source_nodes_json)
                """
                # Get raw result
                result = handle_query(repo, mode, query)

                # Extract response text
                if "error" in result:
                    response_text = f"Error: {result['error']}"
                    source_nodes = {"error": result["error"]}
                else:
                    response_text = result.get("response", "No response available")
                    source_nodes = result.get("source_nodes", [])

                return response_text, source_nodes

            # Wire up events

            # Handle repository selection from dropdown
            repo_dropdown.change(
                fn=handle_repo_selection,
                inputs=[repo_dropdown],
                outputs=[repo_dropdown, selected_repo_textbox, query_btn],
                show_api=False,
            )

            # Handle refresh button - resets to dropdown view
            refresh_repos_btn.click(
                fn=reset_repo_selection,
                outputs=[repo_dropdown, selected_repo_textbox, query_btn],
                show_api=False,
            )

            # Also provide API endpoint for listing repositories
            refresh_repos_btn.click(
                fn=get_available_docs_repo,
                outputs=[repo_dropdown],
                api_name="list_available_docs",
            )

            # Query button uses the textbox value (not dropdown)
            query_btn.click(
                fn=make_query,
                inputs=[
                    selected_repo_textbox,
                    query_mode,
                    query_input,
                ],  # Use textbox, not dropdown
                outputs=[response_output, sources_output],
                api_name="query_documentation",
            )

            # Also allow Enter key to trigger query
            query_input.submit(
                fn=make_query,
                inputs=[
                    selected_repo_textbox,
                    query_mode,
                    query_input,
                ],  # Use textbox, not dropdown
                outputs=[response_output, sources_output],
                show_api=False,
            )

        # ================================
        # Tab 3: Repository Management
        # ================================
        with gr.TabItem("🗂️ Repository Management", visible=ENABLE_REPO_MANAGEMENT):
            gr.Markdown(
                "Manage your ingested repositories - view details and delete repositories when needed."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Repository Statistics")
                    stats_display = gr.JSON(
                        label="Database Statistics",
                        value={"message": "Click refresh to load statistics..."},
                    )
                    refresh_stats_btn = gr.Button(
                        "🔄 Refresh Statistics", variant="secondary"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Repository Details")
                    repos_table = gr.Dataframe(
                        headers=["Repository", "Files", "Last Updated"],
                        datatype=["str", "number", "str"],
                        label="Ingested Repositories",
                        interactive=False,
                        wrap=True,
                    )
                    refresh_repos_btn = gr.Button(
                        "🔄 Refresh Repository List", variant="secondary"
                    )

            gr.Markdown("### 🗑️ Delete Repository")
            gr.Markdown(
                "**⚠️ Warning:** This will permanently delete all documents and metadata for the selected repository."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    delete_repo_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Repository to Delete",
                        value=None,
                        interactive=True,
                        allow_custom_value=False,
                    )

                    # Confirmation checkbox
                    confirm_delete = gr.Checkbox(
                        label="I understand this action cannot be undone", value=False
                    )

                    delete_btn = gr.Button(
                        "🗑️ Delete Repository",
                        variant="stop",
                        size="lg",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    deletion_status = gr.Textbox(
                        label="Deletion Status",
                        value="Select a repository and confirm to enable deletion.",
                        interactive=False,
                        lines=6,
                    )

            # Management functions
            def load_repository_stats():
                """Load overall repository statistics"""
                try:
                    stats = get_repository_stats()
                    return stats
                except Exception as e:
                    return {"error": f"Failed to load statistics: {str(e)}"}

            def load_repository_details():
                """Load detailed repository information as a table"""
                try:
                    details = get_repo_details()

                    if not details:
                        return [["No repositories found", 0, "N/A"]]

                    # Format for dataframe
                    table_data = []
                    for repo in details:
                        last_updated = repo.get("last_updated", "Unknown")
                        if hasattr(last_updated, "strftime"):
                            last_updated = last_updated.strftime("%Y-%m-%d %H:%M")
                        elif last_updated != "Unknown":
                            last_updated = str(last_updated)

                        table_data.append(
                            [
                                repo.get("repo_name", "Unknown"),
                                repo.get("file_count", 0),
                                last_updated,
                            ]
                        )

                    return table_data

                except Exception as e:
                    return [["Error loading repositories", 0, str(e)]]

            def update_delete_dropdown():
                """Update the dropdown with available repositories"""
                try:
                    repos = get_available_repositories()
                    return gr.Dropdown(choices=repos, value=None)
                except Exception as e:
                    print(f"Error updating delete dropdown: {e}")
                    return gr.Dropdown(choices=[], value=None)

            def check_delete_button_state(repo_selected, confirmation_checked):
                """Enable/disable delete button based on selection and confirmation"""
                if repo_selected and confirmation_checked:
                    return gr.Button(interactive=True)
                else:
                    return gr.Button(interactive=False)

            def delete_repository(repo_name: str, confirmed: bool):
                """Delete the selected repository"""
                if not repo_name:
                    return (
                        "❌ No repository selected.",
                        gr.Dropdown(choices=[]),
                        gr.Checkbox(value=False),
                    )

                if not confirmed:
                    return (
                        "❌ Please confirm deletion by checking the checkbox.",
                        gr.Dropdown(choices=[]),
                        gr.Checkbox(value=False),
                    )

                try:
                    # Perform deletion
                    result = delete_repository_data(repo_name)

                    # Prepare status message
                    status_msg = result["message"]
                    if result["success"]:
                        status_msg += "\n\n📊 Deletion Summary:"
                        status_msg += f"\n- Vector documents removed: {result['vector_docs_deleted']}"
                        status_msg += f"\n- Repository record deleted: {'Yes' if result['repo_record_deleted'] else 'No'}"
                        status_msg += f"\n\n✅ Repository '{repo_name}' has been completely removed."

                    # Update dropdown (remove deleted repo)
                    updated_dropdown = update_delete_dropdown()

                    # Reset confirmation checkbox
                    reset_checkbox = gr.Checkbox(value=False)

                    return status_msg, updated_dropdown, reset_checkbox

                except Exception as e:
                    error_msg = f"❌ Error deleting repository: {str(e)}"
                    return error_msg, gr.Dropdown(choices=[]), gr.Checkbox(value=False)

            # Wire up management events
            refresh_stats_btn.click(
                fn=load_repository_stats, outputs=[stats_display], show_api=False
            )

            refresh_repos_btn.click(
                fn=load_repository_details, outputs=[repos_table], show_api=False
            )

            # Update delete dropdown when refreshing repos
            refresh_repos_btn.click(
                fn=update_delete_dropdown,
                outputs=[delete_repo_dropdown],
                show_api=False,
            )

            # Enable/disable delete button based on selection and confirmation
            delete_repo_dropdown.change(
                fn=check_delete_button_state,
                inputs=[delete_repo_dropdown, confirm_delete],
                outputs=[delete_btn],
                show_api=False,
            )

            confirm_delete.change(
                fn=check_delete_button_state,
                inputs=[delete_repo_dropdown, confirm_delete],
                outputs=[delete_btn],
                show_api=False,
            )

            # Delete repository
            delete_btn.click(
                fn=delete_repository,
                inputs=[delete_repo_dropdown, confirm_delete],
                outputs=[deletion_status, delete_repo_dropdown, confirm_delete],
                show_api=False,
            )

            # Load data on tab load
            demo.load(fn=load_repository_stats, outputs=[stats_display], show_api=False)

            demo.load(fn=load_repository_details, outputs=[repos_table], show_api=False)

            demo.load(
                fn=update_delete_dropdown,
                outputs=[delete_repo_dropdown],
                show_api=False,
            )

        # ================================
        # Tab 4: GitHub File Search (Hidden API)
        # ================================
        with gr.TabItem("🔍 GitHub File Search", visible=False):
            gr.Markdown("### 🔧 GitHub Repository File Search API")
            gr.Markdown(
                "Pure API endpoints for GitHub file operations - all responses in JSON format"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📋 List Repository Files")

                    # Repository input for file operations
                    api_repo_input = gr.Textbox(
                        label="Repository URL",
                        placeholder="owner/repo or https://github.com/owner/repo",
                        value="",
                        info="GitHub repository to scan",
                    )

                    # Branch selection
                    api_branch_input = gr.Textbox(
                        label="Branch",
                        value="main",
                        placeholder="main",
                        info="Branch to search (default: main)",
                    )

                    # File extensions
                    api_extensions_input = gr.Textbox(
                        label="File Extensions (comma-separated)",
                        value=".md,.mdx",
                        placeholder=".md,.mdx,.txt",
                        info="File extensions to include",
                    )

                    # List files button
                    list_files_btn = gr.Button("📋 List Files", variant="primary")

                with gr.Column():
                    gr.Markdown("#### 📄 Get Single File")

                    # Single file inputs
                    single_repo_input = gr.Textbox(
                        label="Repository URL",
                        placeholder="owner/repo or https://github.com/owner/repo",
                        value="",
                        info="GitHub repository",
                    )

                    single_file_input = gr.Textbox(
                        label="File Path",
                        placeholder="docs/README.md",
                        value="",
                        info="Path to specific file in repository",
                    )

                    single_branch_input = gr.Textbox(
                        label="Branch",
                        value="main",
                        placeholder="main",
                        info="Branch name (default: main)",
                    )

                    # Get single file button
                    get_single_btn = gr.Button(
                        "📄 Get Single File", variant="secondary"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📚 Get Multiple Files")

                    # Multiple files inputs
                    multiple_repo_input = gr.Textbox(
                        label="Repository URL",
                        placeholder="owner/repo or https://github.com/owner/repo",
                        value="",
                        info="GitHub repository",
                    )

                    multiple_files_input = gr.Textbox(
                        label="File Paths (comma-separated)",
                        placeholder="README.md,docs/guide.md,api/overview.md",
                        value="",
                        lines=3,
                        info="Comma-separated list of file paths",
                    )

                    multiple_branch_input = gr.Textbox(
                        label="Branch",
                        value="main",
                        placeholder="main",
                        info="Branch name (default: main)",
                    )

                    # Get multiple files button
                    get_multiple_btn = gr.Button(
                        "📚 Get Multiple Files", variant="secondary"
                    )

            # Single JSON output for all operations
            gr.Markdown("### 📊 API Response")
            api_response_output = gr.JSON(
                label="JSON Response",
                value={
                    "message": "API responses will appear here",
                    "info": "Use the buttons above to interact with GitHub repositories",
                },
            )

            # Pure API Functions (JSON only responses)
            def list_repository_files(
                repo_url: str, branch: str = "main", extensions: str = ".md,.mdx"
            ):
                """
                List all files in a GitHub repository with specified extensions

                Args:
                    repo_url: GitHub repository URL or owner/repo format
                    branch: Branch name to search (default: main)
                    extensions: Comma-separated file extensions (default: .md,.mdx)

                Returns:
                    JSON response with file list and metadata
                """
                try:
                    if not repo_url.strip():
                        return {"success": False, "error": "Repository URL is required"}

                    # Parse extensions list
                    ext_list = [
                        ext.strip() for ext in extensions.split(",") if ext.strip()
                    ]
                    if not ext_list:
                        ext_list = [".md", ".mdx"]

                    # Get files list
                    files, status_message = fetch_repository_files(
                        repo_url=repo_url,
                        file_extensions=ext_list,
                        github_token=os.getenv("GITHUB_API_KEY"),
                        branch=branch,
                    )

                    if files:
                        return {
                            "success": True,
                            "repository": repo_url,
                            "branch": branch,
                            "extensions": ext_list,
                            "total_files": len(files),
                            "files": files,
                            "status": status_message,
                        }
                    else:
                        return {
                            "success": False,
                            "repository": repo_url,
                            "branch": branch,
                            "extensions": ext_list,
                            "total_files": 0,
                            "files": [],
                            "error": status_message or "No files found",
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to list files: {str(e)}",
                        "repository": repo_url,
                        "branch": branch,
                    }

            def get_single_file(repo_url: str, file_path: str, branch: str = "main"):
                """
                Retrieve a single file from GitHub repository

                Args:
                    repo_url: GitHub repository URL or owner/repo format
                    file_path: Path to the file in the repository
                    branch: Branch name (default: main)

                Returns:
                    JSON response with file content and metadata
                """
                try:
                    if not repo_url.strip():
                        return {"success": False, "error": "Repository URL is required"}

                    if not file_path.strip():
                        return {"success": False, "error": "File path is required"}

                    # Parse repo name
                    if "github.com" in repo_url:
                        repo_name = (
                            repo_url.replace("https://github.com/", "")
                            .replace("http://github.com/", "")
                            .strip("/")
                        )
                    else:
                        repo_name = repo_url.strip()

                    # Load single file
                    documents, failed = load_github_files(
                        repo_name=repo_name,
                        file_paths=[file_path.strip()],
                        branch=branch,
                        github_token=os.getenv("GITHUB_API_KEY"),
                    )

                    if documents and len(documents) > 0:
                        doc = documents[0]
                        return {
                            "success": True,
                            "repository": repo_name,
                            "branch": branch,
                            "file_path": file_path,
                            "file_name": doc.metadata.get("file_name", ""),
                            "file_size": len(doc.text),
                            "content": doc.text,
                            "metadata": doc.metadata,
                            "url": doc.metadata.get("url", ""),
                            "raw_url": doc.metadata.get("raw_url", ""),
                        }
                    else:
                        error_msg = f"Failed to retrieve file: {failed[0] if failed else 'File not found or access denied'}"
                        return {
                            "success": False,
                            "repository": repo_name,
                            "branch": branch,
                            "file_path": file_path,
                            "error": error_msg,
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to get single file: {str(e)}",
                        "repository": repo_url,
                        "file_path": file_path,
                        "branch": branch,
                    }

            def get_multiple_files(
                repo_url: str, file_paths_str: str, branch: str = "main"
            ):
                """
                Retrieve multiple files from GitHub repository

                Args:
                    repo_url: GitHub repository URL or owner/repo format
                    file_paths_str: Comma-separated string of file paths
                    branch: Branch name (default: main)

                Returns:
                    JSON response with multiple file contents and metadata
                """
                try:
                    if not repo_url.strip():
                        return {"success": False, "error": "Repository URL is required"}

                    if not file_paths_str.strip():
                        return {"success": False, "error": "File paths are required"}

                    # Parse file paths from comma-separated string
                    file_paths = [
                        path.strip()
                        for path in file_paths_str.split(",")
                        if path.strip()
                    ]

                    if not file_paths:
                        return {
                            "success": False,
                            "error": "No valid file paths provided",
                        }

                    # Parse repo name
                    if "github.com" in repo_url:
                        repo_name = (
                            repo_url.replace("https://github.com/", "")
                            .replace("http://github.com/", "")
                            .strip("/")
                        )
                    else:
                        repo_name = repo_url.strip()

                    # Load multiple files
                    documents, failed = load_github_files(
                        repo_name=repo_name,
                        file_paths=file_paths,
                        branch=branch,
                        github_token=os.getenv("GITHUB_API_KEY"),
                    )

                    # Process successful documents
                    successful_files = []
                    for doc in documents:
                        file_data = {
                            "file_path": doc.metadata.get("file_path", ""),
                            "file_name": doc.metadata.get("file_name", ""),
                            "file_size": len(doc.text),
                            "content": doc.text,
                            "metadata": doc.metadata,
                            "url": doc.metadata.get("url", ""),
                            "raw_url": doc.metadata.get("raw_url", ""),
                        }
                        successful_files.append(file_data)

                    return {
                        "success": True,
                        "repository": repo_name,
                        "branch": branch,
                        "requested_files": len(file_paths),
                        "successful_files": len(successful_files),
                        "failed_files": len(failed),
                        "files": successful_files,
                        "failed_file_paths": failed,
                        "total_content_size": sum(len(doc.text) for doc in documents),
                        "requested_file_paths": file_paths,
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to get multiple files: {str(e)}",
                        "repository": repo_url,
                        "file_paths": file_paths_str,
                        "branch": branch,
                    }

            # Wire up the GitHub file search events - all output to single JSON component
            list_files_btn.click(
                fn=list_repository_files,
                inputs=[api_repo_input, api_branch_input, api_extensions_input],
                outputs=[api_response_output],
                api_name="list_repository_files",
            )

            get_single_btn.click(
                fn=get_single_file,
                inputs=[single_repo_input, single_file_input, single_branch_input],
                outputs=[api_response_output],
                api_name="get_single_file",
            )

            get_multiple_btn.click(
                fn=get_multiple_files,
                inputs=[
                    multiple_repo_input,
                    multiple_files_input,
                    multiple_branch_input,
                ],
                outputs=[api_response_output],
                api_name="get_multiple_files",
            )

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

if __name__ == "__main__":
    demo.launch(mcp_server=True)
