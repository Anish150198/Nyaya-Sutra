import json
import os
import google.generativeai as genai
 
# ==========================================
# 1. INITIALIZE GEMINI API
# ==========================================
# Configure Gemini API with hardcoded key
GEMINI_API_KEY = "AIzaSyC7YuNKwfJ1CRmwXfNM7XG4XhqzatW0KgA"
 
genai.configure(api_key=GEMINI_API_KEY)
 
print("Initializing Gemini models for legal processing...")
 
# Use Gemini Pro for intake (lighter tasks) and Gemini Pro for drafting/review
# You can adjust model names based on your needs
INTAKE_MODEL = "gemini-3-flash-preview"  # Faster, good for structured extraction
DRAFTING_MODEL = "gemini-3-flash-preview"   # More capable for complex drafting
 
print(f"✓ Intake Model: {INTAKE_MODEL}")
print(f"✓ Drafting Model: {DRAFTING_MODEL}")
print("All models configured successfully! 🚀\n")
 
# ==========================================
# 2. INFERENCE FUNCTIONS
# ==========================================
def call_gemini_api(model_name, system_prompt, user_input, max_tokens=8192, temperature=0.01):
    """Generic function to call Gemini API with system and user prompts."""
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        response = model.generate_content(
            user_input,
            generation_config=generation_config
        )
        
        raw_response = response.text.strip()
        
        # JSON Cleaner Safeguard
        cleaned_response = raw_response
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        return cleaned_response.strip()
        
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        raise
 
def call_gemma_model(system_prompt, user_input, max_new_tokens=2048, temperature=0.01):
    """Call Gemini for intake/extraction tasks using Flash model."""
    combined_prompt = f"{system_prompt}\n\nUSER NARRATIVE:\n{user_input}"
    return call_gemini_api(INTAKE_MODEL, system_prompt, combined_prompt, max_new_tokens, temperature)
 
def call_qwen_model(system_prompt, user_input, max_new_tokens=1024, temperature=0.2):
    """Call Gemini for drafting/review tasks using Pro model."""
    return call_gemini_api(DRAFTING_MODEL, system_prompt, user_input, max_new_tokens, temperature)
 
# ==========================================
# 3. THE AGENTIC ASSEMBLY LINE
# ==========================================
def intake_agent(user_story):
    print("🤖 [Intake Agent] Extracting facts with Gemini Flash...")
    system_prompt = """You are an expert Legal Intake Agent for Nyaya-Sahayak.
Your objective is to read a user's raw narrative, determine the type of legal document they need, and extract the key facts into a STRICT JSON format.
 
RULES:
1. Output ONLY valid JSON. No conversational text.
2. Identify the document type based on the narrative.
3. Extract relevant entities as key-value pairs.
4. If critical details are missing to draft the document, list them in "missing_critical_fields".
 
REQUIRED SCHEMA:
{
  "document_type": "string",
  "extracted_facts": {"dynamic_key_1": "value"},
  "missing_critical_fields": ["list of strings"],
  "target_language": "string or null",
  "is_ready_to_draft": boolean,
  "next_clarification_question": "string or null"
}"""
 
    raw_json_output = call_gemma_model(system_prompt, user_story)
    try:
        return json.loads(raw_json_output)
    except json.JSONDecodeError:
        print(f"❌ JSON Parsing Error. Raw output:\n{raw_json_output}")
        return {"is_ready_to_draft": False, "next_clarification_question": "I had trouble understanding that. Could you rephrase?"}
 
def retrieval_agent(doc_type, facts):
    print("🔎 [Retrieval Agent] Pulling legal context...")
    doc_type_upper = doc_type.upper() if doc_type else "UNKNOWN"
    
    if "FIR" in doc_type_upper or "THEFT" in doc_type_upper:
        return "Section 303(2) of the Bharatiya Nyaya Sanhita (BNS), 2023: Whoever commits theft shall be punished..."
    elif "RENTAL" in doc_type_upper or "AGREEMENT" in doc_type_upper:
        return "Standard Indian Leave and License Agreement Template Clauses. Include Rent amount, Security Deposit, and 11-month duration clauses."
    return "General Legal Provisions under Indian Law."
 
def drafting_agent(doc_type, facts, law):
    print("✍️ [Drafting Agent] Synthesizing legal document with Gemini Pro...")
    system_prompt = f"""You are an expert Indian Legal Drafter. Draft a formal {doc_type} based strictly on the provided JSON facts and retrieved BNS/Civil law context.
 
CONSTRAINTS:
1. Use formal, objective, official Indian legal language.
2. For criminal documents, ONLY use BNS sections. Never use IPC.
3. Do not invent facts. Use "[To be determined]" for missing details.
4. Format clearly using Markdown headings and lists.
5. CRITICAL: At the very end of the document, add a horizontal rule (---) and a section titled "### 💡 Recommended Next Steps". List 3 brief, highly actionable next steps for the client (e.g., "Visit the nearest police station to get this stamped", "Gather photo evidence")."""
 
    user_input = f"INPUT FACTS:\n{json.dumps(facts)}\n\nRETRIEVED LAW:\n{law}"
    return call_qwen_model(system_prompt, user_input)
 
def review_agent(draft, law):
    print("⚖️ [Review Agent] Auditing the draft for legal compliance with Gemini Pro...")
    
    system_prompt = """You are a Senior Legal Reviewer auditing a drafted document.
 
CHECKLIST:
1. Is the format correct?
2. If criminal, does it cite the BNS and strictly avoid the IPC?
3. Are there hallucinated facts?
 
Feel free to think through your review step-by-step.
However, on the VERY LAST LINE, you MUST output your final verdict wrapped in tags:
<VERDICT>PASS</VERDICT>
or
<VERDICT>FAIL: [Brief reason]</VERDICT>"""
 
    user_input = f"RETRIEVED LAW:\n{law}\n\nDRAFT:\n{draft}"
    
    evaluation = call_qwen_model(system_prompt, user_input, max_new_tokens=2048, temperature=0.1)
    
    # Parse the specific XML tags
    if "<VERDICT>PASS</VERDICT>" in evaluation:
        return "PASS"
    elif "<VERDICT>FAIL:" in evaluation:
        # Try to extract just the failure reason
        try:
            reason = evaluation.split("<VERDICT>FAIL:")[1].split("</VERDICT>")[0].strip()
            return reason
        except:
            return evaluation.strip()
    else:
        # Fallback if it didn't use the tags properly
        return "Review formatting error. Check terminal logs."
 
# ==========================================
# 4. TERMINAL TESTING ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚨 INITIATING TERMINAL TEST RUN 🚨")
    print("=" * 60)
    
    test_story = "I was at Orion Mall yesterday evening around 8 PM and someone in a black hoodie stole my phone from my table."
    print(f"USER INPUT: {test_story}\n")
    
    plan = intake_agent(test_story)
    print(f"Extracted Plan: {json.dumps(plan, indent=2)}\n")
    
    # Force drafting for the sake of the terminal test
    doc_type = plan.get("document_type", "FIR")
    facts = plan.get("extracted_facts", {})
    
    law = retrieval_agent(doc_type, facts)
    draft = drafting_agent(doc_type, facts, law)
    status = review_agent(draft, law)
    
    print("-" * 60)
    if "PASS" in status:
        print("\n📄 FINAL GENERATED DOCUMENT:\n")
        print(draft)
    else:
        print(f"Pipeline Failed at Review Stage. Reason: {status}")
 
 