"""
Run this script to fix the _generate_summary method indentation in module_service.py
Usage: python fix_indent.py
"""
import re

filepath = "module_service.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find the old _generate_summary method (from @staticmethod before it to the next section comment)
# We'll replace everything from "# SUMMARY GENERATION" section to "# CHUNKING" section

old_pattern = r'(    # [─]+\n    # SUMMARY GENERATION\n    # [─]+\n).*?(    # [─]+\n    # CHUNKING)'

new_method = r'''\1    @staticmethod
    def _generate_summary(full_text, subject, class_level, page_count):
        """Generate comprehensive, detailed module summary using Gemini."""
        genai = get_genai()
        model = genai.GenerativeModel(GEMINI_MODEL)

        text_for_prompt = full_text[:800000]

        prompt = f"""You are an expert Indian education content creator and NCERT/CBSE curriculum specialist.
Analyze this {subject} document for Class {class_level} and create a COMPREHENSIVE, DETAILED study module.

DOCUMENT ({page_count} pages):
{text_for_prompt}

Create a DETAILED JSON module with this EXACT structure:
{{
    "title": "descriptive title for this module",
    "subject": "{subject}",
    "class": "{class_level}",
    "overview": "3-5 sentence detailed summary of what this document covers, its importance, and what students will learn",
    "topics": [
        {{
            "name": "Topic/Chapter name",
            "explanation": "Detailed 4-8 sentence explanation of this topic in simple student-friendly language. Explain the concept thoroughly as if teaching a student. Include WHY this topic matters.",
            "key_points": [
                "Detailed point 1 - not just a phrase, but a complete sentence explaining the concept",
                "Detailed point 2 with specific facts, numbers, or examples",
                "Detailed point 3"
            ],
            "subtopics": ["subtopic 1", "subtopic 2"],
            "formulas": [
                {{
                    "formula": "The actual formula (e.g., F = ma, E = mc2, 2H2 + O2 -> 2H2O)",
                    "meaning": "What each symbol represents and when to use this formula",
                    "example": "A quick numerical example showing how to apply it"
                }}
            ],
            "diagrams_description": [
                "Description of diagram 1: what it shows, labels, and what student should understand from it"
            ],
            "tables": [
                {{
                    "title": "Table title (e.g., Comparison of Metals and Non-metals)",
                    "headers": ["Column 1", "Column 2", "Column 3"],
                    "rows": [
                        ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
                        ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
                    ]
                }}
            ],
            "real_life_applications": [
                "Real world example 1: How this concept is used in daily life or industry",
                "Real world example 2"
            ],
            "misconceptions": [
                {{
                    "wrong": "Common wrong belief students have",
                    "correct": "The correct understanding with explanation"
                }}
            ]
        }}
    ],
    "important_terms": [
        {{
            "term": "Technical term",
            "definition": "Clear 2-3 sentence definition that a student can understand. Include an example if helpful.",
            "example": "Optional: A specific example of this term in action"
        }}
    ],
    "mind_map": {{
        "central_topic": "Main subject/chapter name",
        "branches": [
            {{
                "branch": "Major topic 1",
                "sub_branches": ["Sub-concept A", "Sub-concept B", "Sub-concept C"]
            }},
            {{
                "branch": "Major topic 2",
                "sub_branches": ["Sub-concept D", "Sub-concept E"]
            }}
        ]
    }},
    "learning_objectives": [
        "After studying this module, student will be able to: objective 1",
        "objective 2",
        "objective 3"
    ],
    "formulas_or_rules": [
        "Complete formula/rule 1 with brief meaning",
        "Complete formula/rule 2 with brief meaning"
    ],
    "quick_revision_notes": [
        "One-liner revision point 1",
        "One-liner revision point 2",
        "One-liner revision point 3"
    ],
    "difficulty_level": "easy or medium or hard",
    "estimated_study_time": "X hours",
    "question_types_possible": ["MCQ", "Short Answer", "Long Answer", "Fill in the Blanks", "True/False", "Diagram Based", "Numerical"],
    "total_pages": {page_count}
}}

CRITICAL RULES:
- EXPLAIN every topic in detail - not just list names. Write as if you are TEACHING a student
- Include ALL formulas, equations, reactions found in the document with their meanings
- Create comparison tables wherever two or more things are compared in the document
- Identify common misconceptions students have about these topics
- Give real-life applications for every major concept
- important_terms: give DETAILED definitions (2-3 sentences), not one-word meanings
- If content is in Hindi, keep Hindi text as-is
- Be EXHAUSTIVE - cover every concept, every formula, every definition in the document
- The mind_map should show how all topics connect to each other
- quick_revision_notes: crisp one-liners for last-minute revision
- Output ONLY valid JSON, no markdown, no backticks, no explanation"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=65000,
                response_mime_type="application/json",
            ),
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

\2'''

new_content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)

if new_content == content:
    print("WARNING: Pattern not matched. Trying alternate approach...")
    # Try simpler replacement
    old_start = "    @staticmethod\n    def _generate_summary(full_text, subject, class_level, page_count):"
    # Find any variation
    import textwrap
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if '_generate_summary' in line and 'def ' in line:
            # Find the @staticmethod line before it
            for j in range(i-1, max(i-5, 0), -1):
                if '@staticmethod' in lines[j]:
                    start_idx = j
                    break
            if start_idx is None:
                start_idx = i
        if start_idx and i > start_idx + 5 and '# CHUNKING' in line:
            # Go back to find the comment line
            for j in range(i-1, i-4, -1):
                if '# ─' in lines[j] or '# -' in lines[j]:
                    end_idx = j
                    break
            if end_idx is None:
                end_idx = i
            break
    
    if start_idx and end_idx:
        print(f"Found method at lines {start_idx+1}-{end_idx+1}")
        new_lines = lines[:start_idx]
        new_method_lines = """    @staticmethod
    def _generate_summary(full_text, subject, class_level, page_count):
        \"\"\"Generate comprehensive, detailed module summary using Gemini.\"\"\"
        genai = get_genai()
        model = genai.GenerativeModel(GEMINI_MODEL)

        text_for_prompt = full_text[:800000]

        prompt = f\"\"\"You are an expert Indian education content creator and NCERT/CBSE curriculum specialist.
Analyze this {subject} document for Class {class_level} and create a COMPREHENSIVE, DETAILED study module.

DOCUMENT ({page_count} pages):
{text_for_prompt}

Create a DETAILED JSON module with this EXACT structure:
{{
    "title": "descriptive title for this module",
    "subject": "{subject}",
    "class": "{class_level}",
    "overview": "3-5 sentence detailed summary of what this document covers, its importance, and what students will learn",
    "topics": [
        {{
            "name": "Topic/Chapter name",
            "explanation": "Detailed 4-8 sentence explanation of this topic in simple student-friendly language. Explain the concept thoroughly as if teaching a student. Include WHY this topic matters.",
            "key_points": [
                "Detailed point 1 - not just a phrase, but a complete sentence explaining the concept",
                "Detailed point 2 with specific facts, numbers, or examples",
                "Detailed point 3"
            ],
            "subtopics": ["subtopic 1", "subtopic 2"],
            "formulas": [
                {{
                    "formula": "The actual formula (e.g., F = ma, E = mc2, 2H2 + O2 -> 2H2O)",
                    "meaning": "What each symbol represents and when to use this formula",
                    "example": "A quick numerical example showing how to apply it"
                }}
            ],
            "diagrams_description": [
                "Description of diagram 1: what it shows, labels, and what student should understand from it"
            ],
            "tables": [
                {{
                    "title": "Table title (e.g., Comparison of Metals and Non-metals)",
                    "headers": ["Column 1", "Column 2", "Column 3"],
                    "rows": [
                        ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
                        ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
                    ]
                }}
            ],
            "real_life_applications": [
                "Real world example 1: How this concept is used in daily life or industry",
                "Real world example 2"
            ],
            "misconceptions": [
                {{
                    "wrong": "Common wrong belief students have",
                    "correct": "The correct understanding with explanation"
                }}
            ]
        }}
    ],
    "important_terms": [
        {{
            "term": "Technical term",
            "definition": "Clear 2-3 sentence definition that a student can understand. Include an example if helpful.",
            "example": "Optional: A specific example of this term in action"
        }}
    ],
    "mind_map": {{
        "central_topic": "Main subject/chapter name",
        "branches": [
            {{
                "branch": "Major topic 1",
                "sub_branches": ["Sub-concept A", "Sub-concept B", "Sub-concept C"]
            }},
            {{
                "branch": "Major topic 2",
                "sub_branches": ["Sub-concept D", "Sub-concept E"]
            }}
        ]
    }},
    "learning_objectives": [
        "After studying this module, student will be able to: objective 1",
        "objective 2",
        "objective 3"
    ],
    "formulas_or_rules": [
        "Complete formula/rule 1 with brief meaning",
        "Complete formula/rule 2 with brief meaning"
    ],
    "quick_revision_notes": [
        "One-liner revision point 1",
        "One-liner revision point 2",
        "One-liner revision point 3"
    ],
    "difficulty_level": "easy or medium or hard",
    "estimated_study_time": "X hours",
    "question_types_possible": ["MCQ", "Short Answer", "Long Answer", "Fill in the Blanks", "True/False", "Diagram Based", "Numerical"],
    "total_pages": {page_count}
}}

CRITICAL RULES:
- EXPLAIN every topic in detail - not just list names. Write as if you are TEACHING a student
- Include ALL formulas, equations, reactions found in the document with their meanings
- Create comparison tables wherever two or more things are compared in the document
- Identify common misconceptions students have about these topics
- Give real-life applications for every major concept
- important_terms: give DETAILED definitions (2-3 sentences), not one-word meanings
- If content is in Hindi, keep Hindi text as-is
- Be EXHAUSTIVE - cover every concept, every formula, every definition in the document
- The mind_map should show how all topics connect to each other
- quick_revision_notes: crisp one-liners for last-minute revision
- Output ONLY valid JSON, no markdown, no backticks, no explanation\"\"\"

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=65000,
                response_mime_type="application/json",
            ),
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

""".split('\n')
        new_lines.extend(new_method_lines)
        new_lines.extend(lines[end_idx:])
        new_content = '\n'.join(new_lines)
        print("Fixed using alternate approach!")
    else:
        print(f"ERROR: Could not find method boundaries. start={start_idx}, end={end_idx}")
        exit(1)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"SUCCESS: {filepath} updated with new _generate_summary method!")
print("Now copy to test-generator-backend:")
print("  copy module_service.py test-generator-backend\\app\\services\\module_service.py")
