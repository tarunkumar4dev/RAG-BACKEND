# ─────────────────────────────────────────────────────────────────
# PATCH for 02_extract_ncert_questions.py
# Replace EXTRACTION_PROMPT and insert_questions function
# ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# REPLACE the old EXTRACTION_PROMPT with this one:
# ═══════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are an expert NCERT textbook analyst for Indian school education.

Given the following NCERT textbook content from Class {class_grade} {subject}, Chapter: "{chapter}", 
extract ALL questions, examples, and activities present in this text.

CONTENT:
---
{content}
---

INSTRUCTIONS:
1. Extract EVERY question you find — exercise questions, in-text questions, examples with solutions, activities, HOTS questions.
2. For each question, identify:
   - The exact question text (copy verbatim from the content — do NOT modify or rephrase)
   - Question number as it appears (Q.1, Q 2, Example 5.3, Activity 9.2, etc.)
   - Section number if visible (like 9.1, 9.2, Exercise, In-Text Questions, etc.)
   - Question type: "exercise" | "example" | "intext" | "activity" | "hots" | "diagram"
   - Answer or solution if present in the text (copy verbatim)
   - If the question has MCQ options, list them
   - Estimated difficulty: "easy" | "medium" | "hard"
   - Estimated marks: 1 (MCQ/fill-blank), 2 (short answer), 3 (reasoning), 5 (long answer/numerical)
3. If the text is purely explanatory with NO questions, return an empty array.
4. Do NOT invent questions. Only extract what is ACTUALLY in the text.
5. Keep the original language (English or Hindi as it appears).

SPECIAL HANDLING:

📊 DATA TABLES: If a question contains a data table (common in Statistics, Economics, Accountancy), 
extract it as structured JSON in the "question_table" field:
  "question_table": {{
    "headers": ["x", "10", "20", "30", "40", "50"],
    "rows": [["f", "5", "8", "12", "7", "3"]]
  }}
If there is NO table, set "question_table": null.

📐 DIAGRAMS/FIGURES: If a question references a figure, diagram, graph, or construction:
  - Set "question_type": "diagram"
  - Extract the figure reference in "figure_ref" field (e.g., "Fig. 9.3", "Figure 13.1")
  - If no specific figure number is mentioned but a diagram is needed, set "figure_ref": "diagram required"
  - Keywords that indicate diagrams: "figure", "fig.", "diagram", "draw", "construct", "sketch", 
    "plot the graph", "shown in the", "given graph", "refer to", "as shown"

RESPOND WITH ONLY valid JSON, no markdown, no commentary:
{{
  "questions": [
    {{
      "question_number": "Q.1",
      "question_text": "Exact question text here",
      "question_type": "exercise",
      "section": "Exercise",
      "answer": "Answer if available, else null",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "difficulty": "medium",
      "marks": 2,
      "question_table": null,
      "figure_ref": null
    }},
    {{
      "question_number": "Q.5",
      "question_text": "Find the mean of the following distribution:",
      "question_type": "exercise",
      "section": "Exercise 13.1",
      "answer": null,
      "options": [],
      "difficulty": "medium",
      "marks": 3,
      "question_table": {{
        "headers": ["Class", "0-10", "10-20", "20-30", "30-40"],
        "rows": [["Frequency", "5", "8", "15", "12"]]
      }},
      "figure_ref": null
    }},
    {{
      "question_number": "Q.3",
      "question_text": "In Fig. 9.3, ABCD is a parallelogram. Find angle x.",
      "question_type": "diagram",
      "section": "Exercise 9.1",
      "answer": "x = 60°",
      "options": [],
      "difficulty": "medium",
      "marks": 2,
      "question_table": null,
      "figure_ref": "Fig. 9.3"
    }}
  ]
}}

If no questions found, return: {{"questions": []}}
"""


# ═══════════════════════════════════════════════════════════════════
# REPLACE the old insert_questions function with this one:
# ═══════════════════════════════════════════════════════════════════

def insert_questions(
    supabase,
    questions: List[Dict],
    class_grade: str,
    subject: str,
    chapter: str,
    source_chunk_ids: List[int],
) -> int:
    """Insert extracted questions into ncert_questions table (with table + diagram support)."""
    if not questions:
        return 0

    rows = []
    for q in questions:
        q_text = (q.get("question_text") or "").strip()
        if not q_text or len(q_text) < 5:
            continue

        # Detect diagram questions even if Gemini didn't tag them
        q_type = q.get("question_type", "exercise")
        figure_ref = q.get("figure_ref")

        diagram_keywords = ["figure", "fig.", "fig ", "diagram", "draw", "construct",
                           "sketch", "plot the graph", "given graph", "shown in"]
        if q_type != "diagram" and any(kw in q_text.lower() for kw in diagram_keywords):
            q_type = "diagram"
            # Try to extract figure reference from text
            if not figure_ref:
                import re
                fig_match = re.search(r'(Fig\.?\s*\d+[\.\d]*)', q_text, re.IGNORECASE)
                if fig_match:
                    figure_ref = fig_match.group(1)
                else:
                    figure_ref = "diagram required"

        # Handle question_table
        question_table = q.get("question_table")
        if question_table:
            # Validate table structure
            if isinstance(question_table, dict) and "headers" in question_table:
                question_table = json.dumps(question_table)
            else:
                question_table = None
        else:
            question_table = None

        row = {
            "class_grade": str(class_grade),
            "subject": subject,
            "chapter": chapter,
            "section": q.get("section"),
            "question_number": q.get("question_number"),
            "question_text": q_text,
            "question_type": q_type,
            "answer": q.get("answer"),
            "options": json.dumps(q.get("options") or []),
            "marks": q.get("marks", 2),
            "difficulty": q.get("difficulty", "medium"),
            "source_chunk_id": source_chunk_ids[0] if source_chunk_ids else None,
            "question_table": question_table,
            "figure_ref": figure_ref,
        }
        rows.append(row)

    if not rows:
        return 0

    try:
        inserted = 0
        for i in range(0, len(rows), 50):
            batch = rows[i:i+50]
            try:
                supabase.table("ncert_questions").insert(batch).execute()
                inserted += len(batch)
            except Exception as batch_err:
                # If question_table column doesn't exist yet, retry without it
                err_str = str(batch_err).lower()
                if "question_table" in err_str or "figure_ref" in err_str:
                    logger.warning("question_table/figure_ref column missing — retrying without")
                    for r in batch:
                        r.pop("question_table", None)
                        r.pop("figure_ref", None)
                    supabase.table("ncert_questions").insert(batch).execute()
                    inserted += len(batch)
                else:
                    raise batch_err
        return inserted
    except Exception as e:
        logger.error(f"Insert failed for {subject} {class_grade} {chapter}: {e}")
        inserted = 0
        for row in rows:
            try:
                supabase.table("ncert_questions").insert(row).execute()
                inserted += 1
            except Exception as e2:
                logger.error(f"Single insert failed: {e2}")
        return inserted