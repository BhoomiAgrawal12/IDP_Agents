import pandas as pd
import pypdf
import spacy
import transformers
import torch
import re
import json
from typing import List, Dict, Any
from pypdf import PdfReader
from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint  # Updated import
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# Set environment variable for Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN")

# Load SpaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Handle case where model is not downloaded
    import sys
    print("SpaCy model not found. Please download it manually with:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Step 1: Document Retriever (Search/Fetch)
def retrieve_document(file_path: str) -> str:
    """Retrieve and extract text from a PDF document."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error retrieving document: {e}")
        return ""

# Step 2: OCR + Parsing Layer
def parse_invoices(text: str) -> List[Dict[str, Any]]:
    """Parse the document to extract invoice data based on the OCR-extracted format."""
    invoices = []
    # Split text into invoice sections (assumes invoices start with '# <number>')
    invoice_texts = re.split(r'#\s*(\d+)', text)[1:]  # Capture invoice number and text

    # Pair invoice numbers with their text
    for i in range(0, len(invoice_texts), 2):
        invoice_number = invoice_texts[i].strip()
        inv_text = invoice_texts[i + 1].strip()
        invoice = {"invoice_number": invoice_number}

        # Extract bill_to (e.g., Bill To : Katrina Edelman)
        bill_to_match = re.search(r"Bill To\s*:\s*(.+?)(?=\n|$)", inv_text, re.DOTALL)
        invoice["bill_to"] = bill_to_match.group(1).strip() if bill_to_match else "Unknown"

        # Extract ship_to (e.g., Ship To : Ilopango, San Salvador, El Salvador)
        ship_to_match = re.search(r"Ship To\s*:\s*(.+?)(?=\n|$)", inv_text, re.DOTALL)
        invoice["ship_to"] = ship_to_match.group(1).strip() if ship_to_match else "Unknown"

        # Extract date (e.g., Aug 17 2012, before 'First Class')
        lines = inv_text.split("\n")
        date = "Unknown"
        for line in lines:
            if re.match(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}", line.strip()):
                date = line.strip()
                break
        invoice["date"] = date

        # Extract ship_mode (e.g., First Class, after date, before balance due)
        ship_mode = "Unknown"
        date_index = lines.index(date) if date != "Unknown" else -1
        if date_index != -1 and date_index + 1 < len(lines):
            next_line = lines[date_index + 1].strip()
            if next_line and not re.match(r"\$\d", next_line):
                ship_mode = next_line
        invoice["ship_mode"] = ship_mode

        # Extract balance_due (e.g., $8,014.33, after ship_mode, before 'Item')
        balance_due = "0.00"
        if ship_mode != "Unknown":
            balance_index = lines.index(ship_mode) + 1 if ship_mode in lines else -1
            if balance_index < len(lines):
                next_line = lines[balance_index].strip()
                if re.match(r"\$\d", next_line):
                    balance_due = re.sub(r"[^\d.]", "", next_line)
        invoice["balance_due"] = balance_due

        # Extract items (e.g., Sharp Copy Machine, Color 7 $1,121.67 $7,851.71)
        items = []
        item_section = False
        item_description = ""
        for j, line in enumerate(lines):
            if "Item Quantity Rate Amount" in line or "Item" in line:
                item_section = True
                continue
            if item_section:
                item_match = re.match(r"(.+?)\s+(\d+)\s+\$(\d+,\d+\.\d{2})\s+\$(\d+,\d+\.\d{2})", line.strip())
                if item_match:
                    item_description = item_match.group(1).strip()
                    items.append({
                        "description": item_description,
                        "quantity": item_match.group(2),
                        "rate": item_match.group(3).replace(",", ""),
                        "amount": item_match.group(4).replace(",", "")
                    })
                elif item_description and j < len(lines) - 1 and not re.match(r"\$\d", line.strip()):
                    # Append additional description lines until numeric values
                    if items:
                        items[-1]["description"] += f"; {line.strip()}"
                if re.match(r"\$\d", line.strip()) and items:
                    item_section = False

        invoice["items"] = items

        # Extract numeric values after items (subtotal, discount, shipping, total)
        numeric_values = []
        for line in lines[lines.index(items[-1]["amount"].replace(".", "")) + 1:] if items else lines:
            match = re.match(r"\$(\d+,\d+\.\d{2})", line.strip())
            if match:
                numeric_values.append(match.group(1).replace(",", ""))

        invoice["subtotal"] = numeric_values[0] if len(numeric_values) > 0 else "0.00"
        invoice["discount"] = numeric_values[1] if len(numeric_values) > 1 else "0.00"
        invoice["shipping"] = numeric_values[2] if len(numeric_values) > 2 else "0.00"
        invoice["total"] = numeric_values[3] if len(numeric_values) > 3 else "0.00"

        # Extract notes (e.g., Notes : Thanks for your business!)
        notes_match = re.search(r"Notes\s*:\s*(.+?)(?=\n|$)", inv_text, re.DOTALL)
        invoice["notes"] = notes_match.group(1).strip() if notes_match else ""

        # Extract order_id (e.g., Order ID : MX-2012-KE1642039-41138)
        order_id_match = re.search(r"Order ID\s*:\s*(.+?)(?=\n|$)", inv_text)
        invoice["order_id"] = order_id_match.group(1).strip() if order_id_match else "Unknown"

        invoices.append(invoice)

    return invoices

import json
# Step 3: Data Storage (Store in CSV)
def store_invoices_to_csv(invoices: List[Dict[str, Any]], csv_path: str) -> pd.DataFrame:
    """Store extracted invoice data in a CSV file."""
    rows = []
    for inv in invoices:
        # Create a row for the invoice header
        row = {
            "invoice_number": inv["invoice_number"],
            "bill_to": inv["bill_to"],
            "ship_to": inv["ship_to"],
            "date": inv["date"],
            "ship_mode": inv["ship_mode"],
            "balance_due": inv["balance_due"],
            "items": json.dumps(inv["items"]),
            "subtotal": inv["subtotal"],
            "discount": inv["discount"],
            "shipping": inv["shipping"],
            "total": inv["total"],
            "notes": inv["notes"],
            "order_id": inv["order_id"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df

# Step 4: Query-Adaptive Data Extractors
memory = ConversationBufferMemory()

def process_user_query(query: str, csv_path: str) -> List[Dict[str, Any]]:
    """Match user query against CSV data using LLM with memory, pre-filtering, and semantic search."""
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV content: {df.head()}")  # Debug: Check CSV content
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found.")
        return []

    # Load models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    huggingfacehub_api_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not huggingfacehub_api_token:
        raise ValueError("HUGGINGFACE_TOKEN not set.")
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=huggingfacehub_api_token,
        task="text-generation",
        max_new_tokens=150,  # Increased for memory context
        temperature=0.2
    )
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Load memory context
    memory_context = memory.load_memory_variables({})["history"]
    memory_prompt = f"Previous conversation: {memory_context}\nCurrent query: {query}\n"
    available_columns = ["invoice_number", "bill_to", "ship_to", "date", "ship_mode",
                        "balance_due", "items", "subtotal", "discount", "shipping",
                        "total", "notes", "order_id"]
    # Use LLM to determine relevant columns with memory
    llm_prompt = f"""{memory_prompt}
    Given the current user query and previous conversation,
    and the available columns in a dataset: invoice_number, bill_to, ship_to, date, ship_mode,
    balance_due, items, subtotal, discount, shipping, total, notes, order_id,
    identify the columns that are most relevant to answer the query.
    - 'names of persons' refers to 'bill_to'.
    - 'Invoice greater than' or similar phrases refer to 'total' or 'subtotal'.
    - Consider feedback from previous interactions (e.g., 'missing names' should prioritize 'bill_to').
    - Return only the column names as a comma-separated list, e.g., 'bill_to, total'."""
    llm_response = llm.invoke(llm_prompt)
    print(f"Raw LLM response: {llm_response}")  # Debug: Check raw LLM output
    relevant_columns = re.findall(r'\b\w+\b', llm_response.lower()) if llm_response else []
    relevant_columns = [col for col in relevant_columns if col in available_columns]
    print(f"LLM identified columns: {relevant_columns}")  # Debug: Check processed columns
    if "names" in query.lower() and "greater" in query.lower():
        relevant_columns = ["bill_to", "total"]
    if not relevant_columns:
        relevant_columns = ["bill_to", "total", "subtotal"]
    if "date" in query.lower() or "purchase date" in query.lower() or "date" in memory_context.lower():
        if "date" not in relevant_columns:
            relevant_columns.append("date")
    # Pre-filter rows based on amount condition
    filtered_df = df.copy()
    amount_condition = re.search(r"greater than (\d+\.?\d*)", query.lower())
    if amount_condition and "total" in relevant_columns:
        threshold = float(amount_condition.group(1))
        filtered_df = filtered_df[filtered_df["total"].astype(float) > threshold]
    elif amount_condition and "subtotal" in relevant_columns:
        threshold = float(amount_condition.group(1))
        filtered_df = filtered_df[filtered_df["subtotal"].astype(float) > threshold]
    print(f"Filtered DF shape: {filtered_df.shape}")  # Debug: Check filtered rows

    # Prepare query embedding
    query_embedding = embedding_model.encode(query.lower(), convert_to_tensor=True)

    relevant_invoices = []
    for _, row in filtered_df.iterrows():
        invoice_text_parts = []
        for col in relevant_columns:
            if col in row and pd.notna(row[col]):
                if col == "items":
                    items = json.loads(row[col]) if pd.notna(row[col]) else []
                    for item in items:
                        invoice_text_parts.append(f"{col}: {item.get('description', '')} Qty: {item.get('quantity', '')} Rate: ${item.get('rate', '')} Amount: ${item.get('amount', '')}")
                else:
                    invoice_text_parts.append(f"{col}: ${row[col]}" if col in ["subtotal", "discount", "shipping", "total", "balance_due"] else f"{col}: {row[col]}")
        invoice_text = " ".join(invoice_text_parts)
        if not invoice_text:
            continue

        invoice_embedding = embedding_model.encode(invoice_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, invoice_embedding).item()
        print(f"Similarity for {row['invoice_number']}: {similarity}")
        if similarity > 1e-8:
            relevant_invoices.append({
                "invoice_number": row["invoice_number"],
                "similarity": similarity,
                "context": invoice_text
            })

    relevant_invoices.sort(key=lambda x: x["similarity"], reverse=True)
    relevant_invoices = relevant_invoices[:10]

    # QA extraction with debug
    final_results = []
    for inv in relevant_invoices:
        result = qa_pipeline(question=query, context=inv["context"])
        print(f"QA score for {inv['invoice_number']}: {result['score']}")
        if result["score"] > 1e-8:
            final_results.append({
                "invoice_number": inv["invoice_number"],
                "answer": result["answer"],
                "context": inv["context"],
                "similarity": inv["similarity"]
            })

    print(f"Relevant invoices: {relevant_invoices}")
    print(f"Final results: {final_results}")
    # Save memory
    memory.save_context({"user_input": query}, {"response": str(final_results)})
    return final_results

# Step 5: Format for Mistral AI
def format_for_mistral(query: str, relevant_invoices: List[Dict[str, Any]]) -> str:
    """Format the retrieved data into a prompt for Mistral AI."""
    context = "\n".join([inv["context"] for inv in relevant_invoices])
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(question=query, context=context)

from langchain.prompts import ChatPromptTemplate
import os

# Step 6: Invoke Mistral AI
def invoke_mistral(query: str, formatted_prompt: str) -> str:
    """Invoke Mistral AI with the formatted prompt."""
    try:
        huggingfacehub_api_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not huggingfacehub_api_token:
            raise ValueError("HUGGINGFACE_TOKEN not set.")
        model = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=huggingfacehub_api_token,
            task="text-generation",
            max_new_tokens=180,
            temperature=1.0
        )
        output_parser = StrOutputParser()
        chain = model | output_parser
        return chain.invoke(formatted_prompt)
    except Exception as e:
        print(f"Error invoking Mistral AI: {e}")
        return "Unable to generate response due to an error."

# Step 7: Quality Control Module
def quality_control(response: str, query: str) -> bool:
    """Check if the response is relevant to the query."""
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    return len(query_words.intersection(response_words)) > len(query_words) // 2

# Step 7: Privacy Module (Optional De-identification)
def apply_privacy(dataset: pd.DataFrame) -> pd.DataFrame:
    """Optionally de-identify sensitive entities (e.g., PERSON names)."""
    # Simulated de-identification (replace PERSON entities with placeholders)
    dataset["source_entity"] = dataset["source_entity"].apply(
        lambda x: "REDACTED" if nlp(x).ents and nlp(x).ents[0].label_ == "PERSON" else x
    )
    dataset["target_entity"] = dataset["target_entity"].apply(
        lambda x: "REDACTED" if nlp(x).ents and nlp(x).ents[0].label_ == "PERSON" else x
    )
    return dataset

def collect_user_feedback(response: str, query: str, relevant_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect user feedback and refine results using memory."""
    max_attempts = 3
    attempt = 1
    current_response = response
    current_invoices = relevant_invoices
    original_query = query
    all_results = current_invoices.copy()  # Track all results for final CSV
    queries_processed = [original_query]  # Track all queries processed for better CSV generation

    # New: Track which results came from which query
    query_results_map = {original_query: current_invoices.copy()}

    while attempt <= max_attempts:
        print(f"\nResponse (Attempt {attempt}):")
        print(current_response)
        feedback = input("Is this response satisfactory? (yes/no, or provide feedback e.g., 'missing names', 'wrong totals', OR enter a completely new query): ").lower()

        if feedback in ["yes", "y"]:
            return {
                "feedback": "Satisfactory",
                "response": current_response,
                "all_results": all_results,
                "queries_processed": queries_processed,
                "query_results_map": query_results_map  # Return the mapping of queries to results
            }
        elif feedback in ["no", "n"] or any(keyword in feedback for keyword in ["missing", "wrong", "incorrect"]):
            print("Feedback noted. Refining response...")
            refined_query = original_query
            if feedback not in ["no", "n"]:
                refined_query += f" {feedback}"  # Append feedback to query
            memory.save_context({"user_input": original_query}, {"feedback": feedback})
            # Re-run query with refined parameters
            refined_results = process_user_query(refined_query, "invoices.csv")
            queries_processed.append(refined_query)
            if refined_results:
                current_response = "\n".join([f"{res['answer']}: ${res['context'].split('total: $')[1].split(' ')[0] if 'total: $' in res['context'] else 'N/A'} (Date: {res['context'].split('date: ')[1].split(' ')[0] if 'date:' in res['context'] else 'N/A'})" for res in refined_results])
                current_invoices = refined_results
                all_results.extend(refined_results)  # Add new results to all_results

                # New: Track results for this refined query
                query_results_map[refined_query] = refined_results.copy()

                if quality_control(current_response, refined_query):  # Assuming quality_control is defined
                    attempt += 1
                    continue
            print("No further improvements possible with current data.")
            return {
                "feedback": feedback,
                "response": current_response,
                "all_results": all_results,
                "queries_processed": queries_processed,
                "query_results_map": query_results_map
            }
        else:
            # This is the key change: treat input as a completely new query when it doesn't look like feedback
            new_query = feedback  # The feedback is actually a new query
            print(f"Processing new query: '{new_query}'")

            # Save context between queries
            memory.save_context({"user_input": original_query}, {"new_query": new_query})

            # Execute the completely new query
            new_results = process_user_query(new_query, "invoices.csv")
            queries_processed.append(new_query)

            if new_results:
                new_response = "\n".join([f"{res['answer']}: ${res['context'].split('total: $')[1].split(' ')[0] if 'total: $' in res['context'] else 'N/A'} (Date: {res['context'].split('date: ')[1].split(' ')[0] if 'date:' in res['context'] else 'N/A'})" for res in new_results])
                current_response = new_response
                current_invoices = new_results
                all_results.extend(new_results)  # Add new results to all_results

                # New: Track results for this new query
                query_results_map[new_query] = new_results.copy()

                if quality_control(current_response, new_query):
                    attempt += 1
                    continue
            else:
                print("No results found for the new query.")
                current_response = "No relevant invoices found matching your query."

            attempt += 1

    print("Max feedback attempts reached.")
    return {
        "feedback": "Unsatisfactory after max attempts",
        "response": current_response,
        "all_results": all_results,
        "queries_processed": queries_processed,
        "query_results_map": query_results_map
    }

# Step 9: Continuous Learning Update
def update_learning(feedback: Dict[str, Any]) -> None:
    """Update system based on feedback."""
    if "incorrect" in feedback["feedback"].lower():
        print("Noted: Will adjust query matching logic.")
    elif "missing" in feedback["feedback"].lower():
        print("Noted: Will improve invoice data extraction.")
    elif "satisfactory" not in feedback["feedback"].lower():
        print("Noted: Will review overall system performance.")
    else:
        print("Feedback received. System performance is satisfactory.")

# Step 10: Final Delivery
def deliver_response(response: str) -> None:
    """Deliver the final response to the user."""
    print("\nFinal Response:")
    print(response)

# Step 11: CSV Download (New Step)
def download_relevant_data(relevant_invoices: List[Dict[str, Any]], filename: str = "relevant_invoices.csv", query: str = ""):
    """Create and provide a download link for relevant invoice data."""
    if not relevant_invoices:
        print("No relevant invoices to download.")
        return

    # Create a better filename based on the query if provided
    if query:
        # Replace non-alphanumeric characters with underscores
        query_name = re.sub(r'[^a-zA-Z0-9]', '_', query)
        # Limit length
        query_name = query_name[:30]
        # Add to filename
        base_filename = f"{query_name}_results"
    else:
        base_filename = "relevant_invoices"

    filename = f"{base_filename}.csv"

    # Prepare data for CSV - improve context parsing
    data = []
    for inv in relevant_invoices:
        context_dict = {}
        # Better parsing of context
        if "context" in inv:
            context_parts = inv["context"].split()
            current_key = None
            current_value = []

            for part in context_parts:
                if ":" in part and part.split(":")[0] in ["invoice_number", "bill_to", "ship_to", "date", "ship_mode",
                                                       "balance_due", "subtotal", "discount", "shipping", "total"]:
                    # Save previous key-value pair
                    if current_key:
                        context_dict[current_key] = " ".join(current_value)
                    # Start new key-value pair
                    current_key = part.split(":")[0]
                    current_value = [part.split(":", 1)[1]] if len(part.split(":", 1)) > 1 else []
                elif current_key:
                    current_value.append(part)

            # Save the last key-value pair
            if current_key and current_value:
                context_dict[current_key] = " ".join(current_value)

        # Add basic invoice data
        data_row = {
            "invoice_number": inv.get("invoice_number", ""),
            "answer": inv.get("answer", ""),
            "similarity": inv.get("similarity", 0),
        }

        # Add extracted context fields
        for key, value in context_dict.items():
            data_row[key] = value.strip("$") if key in ["total", "subtotal", "balance_due", "shipping", "discount"] else value

        data.append(data_row)

    # Create main CSV with all data
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Relevant data saved as {filename}. Download it from your environment.")

    # Create a themed summary CSV if appropriate columns exist
    # For example, for queries about names and totals
    if "bill_to" in df.columns and "total" in df.columns:
        summary_columns = ["invoice_number", "bill_to", "total"]
        if "date" in df.columns:
            summary_columns.append("date")
        summary_df = df[summary_columns].copy()
        summary_filename = f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary data saved as {summary_filename}.")

    # Try to download in Colab
    try:
        from google.colab import files
        files.download(filename)
    except ImportError:
        print("To download, manually retrieve the file from your working directory.")

    return filename

# Main Workflow
def agentic_ai_workflow(file_path: str, user_query: str) -> None:
    """Execute the full Agentic AI workflow."""
    # Step 1: Retrieve document
    raw_text = retrieve_document(file_path)
    if not raw_text:
        print("Failed to retrieve document.")
        return

    # Step 2: Parse invoices
    invoices = parse_invoices(raw_text)
    print("invoices", invoices)
    # Step 3: Store invoices in CSV
    csv_path = "invoices.csv"
    store_invoices_to_csv(invoices, csv_path)

    # Step 4: Process user query
    relevant_invoices = process_user_query(user_query, csv_path)
    if not relevant_invoices:
        print("No relevant invoices found for the query.")
        return
    print("relevant_invoices", relevant_invoices)

    # Track query-specific results
    query_results_map = {user_query: relevant_invoices.copy()}

    # Step 5: Format for Mistral AI
    formatted_prompt = format_for_mistral(user_query, relevant_invoices)

    # Step 6: Invoke Mistral AI
    response = invoke_mistral(user_query, formatted_prompt)

    # Step 7: Quality control
    if not quality_control(response, user_query):
        print("Initial response may not be relevant. Collecting feedback to refine.")

    # Step 8: Collect user feedback
    feedback = collect_user_feedback(response, user_query, relevant_invoices)

    # Extract all queries that were processed
    queries_processed = feedback.get("queries_processed", [user_query])

    # Get the mapping of which results came from which query
    query_results_map = feedback.get("query_results_map", {user_query: relevant_invoices})

    # Step 9: Update learning
    update_learning(feedback)

    # Step 10: Deliver final response
    deliver_response(feedback["response"])

    # Enhanced Step 11: Download relevant data with all collected results
    all_results = feedback.get("all_results", relevant_invoices)

    # Save main results file with all collected results
    main_filename = download_relevant_data(all_results, query=user_query)

    # If multiple queries were processed, save individual query results using the map
    if len(queries_processed) > 1:
        print("\nGenerating query-specific CSV files:")

        for query, results in query_results_map.items():
            if query == user_query:  # Skip main query as we already saved it
                continue

            if results:
                print(f"  - Creating results for: '{query}'")
                download_relevant_data(results, query=query)

    print("\nDocument processing and query handling complete.")

# Example Usage
file_path = "/content/10.pdf"  # Replace with actual PDF path
user_query = "give me all the name of purchasers"

agentic_ai_workflow(file_path, user_query)

