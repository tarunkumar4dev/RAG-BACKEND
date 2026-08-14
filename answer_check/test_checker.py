import google.genai as genai
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============ CONFIG ============
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL = "gemini-2.5-flash"
# Poppler path — adjust if different on your system
POPPLER_PATH = r"C:\Users\login\Downloads\RAG-BACKEND\ncert-ingestion\poppler-24.08.0\Library\bin"
# ================================

client = genai.Client(api_key=API_KEY)


def pdf_to_images(pdf_path: str, output_dir: str = "sheets") -> list:
    """Convert PDF pages to images and return list of image paths."""
    from pdf2image import convert_from_path
    
    print(f"  Converting PDF to images: {pdf_path}")
    images = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        dpi=200,  # Good balance of quality vs size
        fmt="jpeg"
    )
    
    pdf_name = Path(pdf_path).stem
    image_paths = []
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.jpg")
        img.save(img_path, "JPEG", quality=90)
        image_paths.append(img_path)
        print(f"    Saved page {i+1}: {img_path}")
    
    print(f"  Total pages converted: {len(images)}")
    return image_paths


def load_answer_key(path="answer_key.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_sheet(image_paths: list, answer_key: dict, difficulty: str = "medium"):
    """
    Send answer sheet images + answer key to Gemini and get marks.
    
    difficulty: easy / medium / hard / extreme
    """
    
    difficulty_prompts = {
        "easy": "Be lenient in checking. If the student shows basic understanding of the concept, award partial marks generously. Even if exact words don't match, give 40-50% marks if the core idea is present.",
        "medium": "Follow standard CBSE marking scheme. Award step marks as defined. Partial marks for partially correct answers. Be fair but not overly strict.",
        "hard": "Be strict. Key terms must be present. Answers should be complete and well-structured. Deduct marks for vague or incomplete answers. Require 70-80% keyword matching.",
        "extreme": "Be very strict. Require near-perfect answers with correct terminology, grammar, complete sentences, and all key points covered. 90-95% accuracy expected."
    }
    
    prompt = f"""You are an experienced Indian school teacher checking answer sheets.

CHECKING STRICTNESS: {difficulty.upper()}
{difficulty_prompts.get(difficulty, difficulty_prompts["medium"])}

ANSWER KEY:
{json.dumps(answer_key["questions"], indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Look at the student's answer sheet image(s) carefully
2. Identify each question number and the student's written answer
3. Compare each answer with the answer key
4. Award marks based on the marking scheme and checking strictness level
5. If you cannot read a particular answer clearly, mention that in feedback

IMPORTANT:
- If a question number is missing or you can't find it, mark it as 0 with feedback "Answer not found/not readable"
- For questions where student wrote extra points beyond the answer key, still evaluate fairly
- Consider common spelling mistakes in student handwriting — don't penalize minor spelling errors unless strictness is "extreme"

Respond ONLY in this exact JSON format, nothing else:
{{
  "student_answers": [
    {{
      "q_no": 1,
      "extracted_answer": "What you read from the student's sheet (exact text as best as you can read)",
      "marks_awarded": 2,
      "max_marks": 3,
      "confidence": "high/medium/low",
      "feedback": "Brief reason for marks given"
    }}
  ],
  "total_marks": 18,
  "max_marks": 30,
  "overall_remarks": "Brief overall assessment",
  "readability_score": "good/average/poor"
}}"""

    # Upload images
    uploaded_files = []
    for img_path in image_paths:
        print(f"  Uploading {img_path}...")
        uploaded = client.files.upload(file=img_path)
        uploaded_files.append(uploaded)
    
    # Build content parts
    contents = []
    for uf in uploaded_files:
        contents.append(uf)
    contents.append(prompt)
    
    print("  Sending to Gemini... (this may take 30-60 seconds)")
    
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
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
        return {"raw_response": response.text}


def run_full_test():
    """Run the test on all sheets and save results."""
    
    answer_key = load_answer_key()
    sheets_dir = Path("sheets")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # ============================================
    # STEP 1: Convert any PDFs to images first
    # ============================================
    pdf_files = list(sheets_dir.glob("*"))
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s) — converting to images...\n")
        for pdf_file in pdf_files:
            # Skip question_paper.pdf — only process answer sheets
            if "question" in pdf_file.stem.lower():
                print(f"  Skipping {pdf_file.name} (question paper, not answer sheet)")
                continue
            pdf_to_images(str(pdf_file), str(sheets_dir))
        print()
    
    # ============================================
    # STEP 2: Collect all images (original + converted)
    # ============================================
    sheet_groups = {}
    for img_file in sorted(sheets_dir.glob("*.*")):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            # Group by sheet name
            # For PDFs converted: answer_sheet_page_1.jpg → group "answer_sheet"
            # For direct images: sheet_01_page_1.jpg → group "01"
            parts = img_file.stem.rsplit("_page_", 1)
            if len(parts) == 2:
                sheet_name = parts[0]
            else:
                # Try old naming: sheet_01_page_1
                parts2 = img_file.stem.split("_")
                if len(parts2) >= 2:
                    sheet_name = parts2[0] + "_" + parts2[1]
                else:
                    sheet_name = img_file.stem
            
            if sheet_name not in sheet_groups:
                sheet_groups[sheet_name] = []
            sheet_groups[sheet_name].append(str(img_file))
    
    if not sheet_groups:
        print("ERROR: No answer sheets found!")
        print("Put PDFs or images in the sheets/ folder.")
        print("  PDFs: answer_sheet.pdf (will auto-convert to images)")
        print("  Images: sheet_01_page_1.jpg, sheet_01_page_2.jpg, etc.")
        return
    
    print(f"Found {len(sheet_groups)} answer sheet(s) to check\n")
    for name, paths in sheet_groups.items():
        print(f"  {name}: {len(paths)} page(s)")
    print()
    
    # ============================================
    # STEP 3: Check each sheet
    # ============================================
    all_results = {}
    difficulties = ["medium"]  # Test with medium first
    
    for difficulty in difficulties:
        print(f"=== Checking with difficulty: {difficulty.upper()} ===\n")
        
        for sheet_name, image_paths in sorted(sheet_groups.items()):
            print(f"Sheet '{sheet_name}' ({len(image_paths)} pages):")
            
            start_time = time.time()
            result = check_sheet(image_paths, answer_key, difficulty)
            elapsed = time.time() - start_time
            
            print(f"  Time taken: {elapsed:.1f} seconds")
            
            if "total_marks" in result:
                print(f"  Marks: {result['total_marks']}/{result['max_marks']}")
                print(f"  Readability: {result.get('readability_score', 'N/A')}")
                
                for qa in result.get("student_answers", []):
                    conf = qa.get('confidence', '?')
                    print(f"    Q{qa['q_no']}: {qa['marks_awarded']}/{qa['max_marks']} "
                          f"(confidence: {conf}) — {qa.get('feedback', '')[:60]}")
            else:
                print(f"  ERROR: Unexpected response format")
            
            print()
            
            result_key = f"{sheet_name}_{difficulty}"
            all_results[result_key] = {
                "sheet_name": sheet_name,
                "difficulty": difficulty,
                "time_seconds": elapsed,
                "result": result
            }
            
            time.sleep(2)
    
    # ============================================
    # STEP 4: Save results
    # ============================================
    output_path = results_dir / "test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll results saved to {output_path}")
    print("\n=== SUMMARY ===")
    print(f"Sheets checked: {len(sheet_groups)}")
    
    print(f"\n{'Sheet':<25} {'Marks':<12} {'Readability':<12} {'Time':<10}")
    print("-" * 60)
    for key, data in all_results.items():
        r = data["result"]
        marks = f"{r.get('total_marks', '?')}/{r.get('max_marks', '?')}"
        readability = r.get("readability_score", "?")
        time_taken = f"{data['time_seconds']:.1f}s"
        print(f"{data['sheet_name']:<25} {marks:<12} {readability:<12} {time_taken:<10}")


if __name__ == "__main__":
    print("=" * 50)
    print("  a4ai Answer Sheet Checker — Manual Test")
    print("=" * 50)
    print()
    run_full_test()