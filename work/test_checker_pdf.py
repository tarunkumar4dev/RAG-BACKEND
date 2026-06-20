import google.genai as genai
import json
import time
import os
from pathlib import Path

# ============ CONFIG ============
API_KEY = "AIzaSyD8z4kiwpNBdyBv9uYX9iTda6fRbqTL9dc"
MODEL = "gemini-2.5-flash" # Supports PDF natively
# ================================

client = genai.Client(api_key=API_KEY)

def load_answer_key(path="answer_key.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_sheet(pdf_path: str, answer_key: dict, difficulty: str = "medium"):
    """
    Send answer sheet PDF + answer key to Gemini and get marks.
    """
    
    difficulty_prompts = {
        "easy": "Be lenient. Award partial marks generously. If core idea is present, give 40-50% marks.",
        "medium": "Follow standard CBSE marking scheme. Be fair but not overly strict.",
        "hard": "Be strict. Key terms must be present. Deduct for vague answers.",
        "extreme": "Very strict. Near-perfect answers only. 90-95% accuracy expected."
    }
    
    # Build the prompt
    prompt = f"""You are an experienced Indian school teacher checking answer sheets.

CHECKING STRICTNESS: {difficulty.upper()}
{difficulty_prompts.get(difficulty, difficulty_prompts["medium"])}

ANSWER KEY:
{json.dumps(answer_key["questions"], indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Read the student's answer sheet PDF carefully
2. Identify each question number and the student's written answer
3. Compare each answer with the answer key
4. Award marks based on the marking scheme and checking strictness level
5. If you cannot read a particular answer clearly, mention that in feedback

IMPORTANT:
- For handwritten answers, try your best to read them. Note any illegible parts.
- If a question number is missing or can't be found, mark 0 with feedback "Answer not found"
- Don't penalize minor spelling mistakes (unless extreme strictness)
- For questions with diagrams, evaluate based on labels and explanation

Respond ONLY in this exact JSON format:
{{
  "student_answers": [
    {{
      "q_no": 1,
      "extracted_answer": "What you read from student's sheet",
      "marks_awarded": 2,
      "max_marks": 3,
      "confidence": "high/medium/low",
      "feedback": "Brief reason for marks given",
      "legibility_issue": false
    }}
  ],
  "total_marks": 18,
  "max_marks": 30,
  "overall_remarks": "Brief overall assessment",
  "readability_score": "good/average/poor"
}}"""

    print(f"  Uploading PDF: {pdf_path}")
    
    # Upload PDF directly
    pdf_file = client.files.upload(file=pdf_path)
    
    print("  Sending to Gemini... (30-60 seconds)")
    
    response = client.models.generate_content(
        model=MODEL,
        contents=[pdf_file, prompt],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    )
    
    # Parse response
    try:
        result = json.loads(response.text)
        return result
    except json.JSONDecodeError:
        print("  WARNING: Could not parse JSON. Raw response:")
        print(response.text[:1000])
        return {"raw_response": response.text, "error": "JSON parse failed"}

def run_full_test():
    """Run the test on all PDF sheets."""
    
    answer_key = load_answer_key()
    sheets_dir = Path("sheets")  # Put PDFs here
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted(sheets_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("ERROR: No PDF files found in sheets/ folder!")
        print("Put PDFs named like: sheet_01.pdf, sheet_02.pdf")
        return
    
    print(f"Found {len(pdf_files)} answer sheets to check\n")
    
    all_results = {}
    
    # Test with different difficulty levels
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        print(f"\n{'='*50}")
        print(f"Checking with difficulty: {difficulty.upper()}")
        print('='*50)
        
        for pdf_path in pdf_files:
            sheet_name = pdf_path.stem  # e.g., "sheet_01"
            print(f"\n📄 Sheet: {sheet_name}")
            
            start_time = time.time()
            result = check_sheet(str(pdf_path), answer_key, difficulty)
            elapsed = time.time() - start_time
            
            print(f"  ⏱️ Time: {elapsed:.1f} seconds")
            
            if "total_marks" in result:
                print(f"  📊 Marks: {result['total_marks']}/{result['max_marks']}")
                print(f"  📖 Readability: {result.get('readability_score', 'N/A')}")
                
                # Print per-question breakdown
                for qa in result.get("student_answers", []):
                    legible = "⚠️ illegible" if qa.get('legibility_issue') else ""
                    print(f"    Q{qa['q_no']}: {qa['marks_awarded']}/{qa['max_marks']} "
                          f"(confidence: {qa.get('confidence', '?')}) {legible}")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unexpected response')}")
            
            # Save result
            result_key = f"{sheet_name}_{difficulty}"
            all_results[result_key] = {
                "sheet": sheet_name,
                "difficulty": difficulty,
                "time_seconds": elapsed,
                "result": result
            }
            
            # Delay to avoid rate limits
            time.sleep(2)
    
    # Save all results
    output_path = results_dir / "test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"✅ All results saved to {output_path}")
    print('='*50)
    
    # Print summary table
    print("\n📈 SUMMARY TABLE:")
    print(f"{'Sheet':<15} {'Difficulty':<10} {'Marks':<12} {'Time':<10}")
    print("-" * 50)
    for key, data in all_results.items():
        r = data["result"]
        marks = f"{r.get('total_marks', '?')}/{r.get('max_marks', '?')}"
        time_taken = f"{data['time_seconds']:.1f}s"
        print(f"{data['sheet']:<15} {data['difficulty']:<10} {marks:<12} {time_taken:<10}")

def test_single_pdf():
    """Quick test with just one PDF."""
    answer_key = load_answer_key()
    
    pdf_path = input("Enter PDF filename (from sheets/ folder): ")
    pdf_path = Path("sheets") / pdf_path
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"\nTesting: {pdf_path}")
    result = check_sheet(str(pdf_path), answer_key, "medium")
    
    print("\n" + "="*50)
    print("RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("  a4ai Answer Sheet Checker — PDF Version")
    print("="*50)
    print("\nOptions:")
    print("  1. Run full test (all PDFs in sheets/ folder)")
    print("  2. Test single PDF")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_single_pdf()
    else:
        run_full_test()