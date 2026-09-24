import os
import re
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pymupdf
from supabase import create_client

# Ensure UTF-8 stdout and unbuffered output on Windows
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# Load environment
load_dotenv("c:/Users/login/Downloads/RAG-BACKEND/ncert-ingestion/.env")
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

from app.core.db_pool import get_db_connection

BUCKET = "ncert-diagrams"
FIG_CAPTION_PATTERN = re.compile(r"^\s*(?:figure|fig\.?)\s*(\d+\.\d+)\s*(.*)", re.IGNORECASE)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_") or "unnamed"


def process_chapter_pdf(pdf_path: Path, class_grade: str, subject: str, chapter_name: str):
    print(f"\n==================================================")
    print(f"[PDF] Processing: {chapter_name} ({pdf_path.name})")
    print(f"      Class: {class_grade} | Subject: {subject}")
    
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        print(f"   [ERROR] Cannot open {pdf_path.name}: {e}")
        return 0, 0

    print(f"      Total Pages: {len(doc)}")
    
    # Step 1: Detect figures and captions
    figures = {}
    for page_idx, page in enumerate(doc):
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                m = FIG_CAPTION_PATTERN.match(line_text)
                if m:
                    fig_num = m.group(1)
                    desc = m.group(2).strip()
                    bbox = line.get("bbox")
                    if fig_num not in figures or len(desc) > len(figures[fig_num]["desc"]):
                        figures[fig_num] = {
                            "page_num": page_idx + 1,
                            "page_idx": page_idx,
                            "fig_num": fig_num,
                            "desc": desc,
                            "caption_bbox": bbox,
                        }
                        
    print(f"      Detected {len(figures)} Figure Captions.")
    if not figures:
        doc.close()
        return 0, 0
        
    conn = get_db_connection()
    cur = conn.cursor()
    
    diagrams_uploaded = 0
    questions_linked = 0
    
    clean_chap = sanitize_filename(chapter_name)
    clean_subj = sanitize_filename(subject)
    
    for fig_num, info in sorted(figures.items(), key=lambda x: [int(p) for p in x[0].split('.') if p.isdigit()]):
        page = doc[info["page_idx"]]
        cap_bbox = pymupdf.Rect(info["caption_bbox"])
        
        # Gather vector drawings and images above or around the caption
        drawings = page.get_drawings()
        d_rects = [d["rect"] for d in drawings if d.get("rect") and d["rect"].y1 <= cap_bbox.y1 + 5 and d["rect"].y0 >= 40]
        
        img_list = page.get_images()
        img_rects = []
        for img in img_list:
            try:
                r = page.get_image_bbox(img)
                if r.y1 <= cap_bbox.y1 + 5 and r.y0 >= 40:
                    img_rects.append(r)
            except Exception:
                pass
                
        all_rects = d_rects + img_rects
        if not all_rects:
            # Fallback envelope
            clip_rect = pymupdf.Rect(
                max(0, cap_bbox.x0 - 60),
                max(40, cap_bbox.y0 - 280),
                min(page.rect.width, cap_bbox.x1 + 60),
                min(page.rect.height, cap_bbox.y1 + 10)
            )
        else:
            min_x = max(0, min(r.x0 for r in all_rects) - 8)
            min_y = max(40, min(r.y0 for r in all_rects) - 8)
            max_x = min(page.rect.width, max(r.x1 for r in all_rects) + 8)
            max_y = min(page.rect.height, max(cap_bbox.y1 + 8, max(r.y1 for r in all_rects) + 8))
            clip_rect = pymupdf.Rect(min_x, min_y, max_x, max_y)
            
        try:
            pix = page.get_pixmap(clip=clip_rect, dpi=150, alpha=False)
            img_bytes = pix.tobytes("png")
        except Exception as e:
            print(f"   [ERROR] Render failed for Fig {fig_num}: {e}")
            continue
            
        storage_path = f"class{class_grade}/{clean_subj}/{clean_chap}/fig_{fig_num.replace('.', '_')}.png"
        try:
            supabase.storage.from_(BUCKET).upload(
                path=storage_path,
                file=img_bytes,
                file_options={"content-type": "image/png", "upsert": "true"}
            )
            public_url = supabase.storage.from_(BUCKET).get_public_url(storage_path)
            diagrams_uploaded += 1
            print(f"   [UPLOAD] Fig {fig_num} ({pix.width}x{pix.height}) -> {storage_path}")
        except Exception as e:
            print(f"   [ERROR] Upload failed for Fig {fig_num}: {e}")
            continue
            
        # Match questions in DB
        candidates = [
            fig_num,
            f"Fig. {fig_num}",
            f"Fig.{fig_num}",
            f"Figure {fig_num}",
            f"Figure. {fig_num}",
        ]
        
        cur.execute("""
            SELECT id, question_number, figure_ref, question_text 
            FROM ncert_questions 
            WHERE class_grade = %s 
              AND LOWER(subject) = LOWER(%s)
              AND chapter ILIKE %s
              AND (
                figure_ref = ANY(%s) 
                OR question_text ILIKE %s 
                OR question_text ILIKE %s
                OR question_text ILIKE %s
              )
        """, (
            str(class_grade), 
            subject, 
            f"%{chapter_name}%", 
            candidates, 
            f"%Fig. {fig_num}%", 
            f"%Figure {fig_num}%", 
            f"%Fig {fig_num}%"
        ))
        
        matched_qs = cur.fetchall()
        for q in matched_qs:
            cur.execute("""
                UPDATE ncert_questions 
                SET image_url = %s, 
                    figure_ref = COALESCE(figure_ref, %s)
                WHERE id = %s
            """, (public_url, f"Fig. {fig_num}", q[0]))
            questions_linked += 1
            print(f"      [LINK] Question ID {q[0]} ({q[1]}) -> Fig {fig_num}")
            
        conn.commit()
        
    cur.close()
    conn.close()
    doc.close()
    
    print(f"   [DONE] {diagrams_uploaded} diagrams uploaded, {questions_linked} questions linked.")
    return diagrams_uploaded, questions_linked


def main():
    parser = argparse.ArgumentParser(description="NCERT Diagram Extractor & Linker")
    parser.add_argument("--class_grade", default="10", help="Class grade (default: 10)")
    parser.add_argument("--subject", default="Science", help="Subject (default: Science)")
    parser.add_argument("--folder", default="Class10_Science", help="PDF Folder name")
    parser.add_argument("--chapter", help="Specific chapter filename (optional)")
    args = parser.parse_args()

    base_pdf_dir = Path("c:/Users/login/Downloads/RAG-BACKEND/ncert-ingestion/pdfs")
    target_dir = base_pdf_dir / args.folder
    
    if not target_dir.exists():
        print(f"[ERROR] Directory not found: {target_dir}")
        return

    if args.chapter:
        pdf_files = [target_dir / args.chapter if not args.chapter.endswith('.pdf') else target_dir / args.chapter]
    else:
        pdf_files = sorted(list(target_dir.glob("*.pdf")))

    print(f"Starting extraction for {len(pdf_files)} PDF(s) in {args.folder}...")
    total_diags = 0
    total_qs = 0
    for p in pdf_files:
        if not p.exists():
            print(f"[WARN] File not found: {p}")
            continue
        chap_name = p.stem
        d_count, q_count = process_chapter_pdf(p, args.class_grade, args.subject, chap_name)
        total_diags += d_count
        total_qs += q_count

    print(f"\n==================================================")
    print(f"ALL COMPLETE: {total_diags} diagrams uploaded, {total_qs} questions linked.")
    print(f"==================================================")


if __name__ == "__main__":
    main()

