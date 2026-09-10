import re

filepath = r"test-generator-backend/app/services/module_service.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

new_method = '''
    @staticmethod
    def _extract_and_upload_images(filepath, module_id, teacher_id):
        fitz = get_fitz()
        sb = get_supabase()
        if not sb:
            return []
        doc = fitz.open(filepath)
        extracted_images = []
        image_index = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if len(image_bytes) < 5000 or width < 80 or height < 80:
                        continue
                    image_filename = f"img_{page_num+1}_{image_index}.{image_ext}"
                    storage_path = f"{teacher_id}/{module_id}/images/{image_filename}"
                    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
                    mime_type = mime_map.get(image_ext.lower(), "image/png")
                    sb.storage.from_("Modules").upload(storage_path, image_bytes, file_options={"content-type": mime_type, "upsert": "true"})
                    public_url = sb.storage.from_("Modules").get_public_url(storage_path)
                    extracted_images.append({"index": image_index, "page": page_num + 1, "url": public_url, "width": width, "height": height})
                    image_index += 1
                except Exception as e:
                    logger.warning(f"Image extract failed: {e}")
                    continue
        doc.close()
        logger.info(f"Extracted {len(extracted_images)} images for module {module_id}")
        return extracted_images

'''

# Insert before _generate_summary
lines = content.split('\n')
insert_idx = None
for i, line in enumerate(lines):
    if '_generate_summary' in line and 'def ' in line:
        for j in range(i-1, max(i-10, 0), -1):
            if '@staticmethod' in lines[j]:
                insert_idx = j
                break
        if insert_idx is None:
            insert_idx = i
        break

if insert_idx:
    new_lines = lines[:insert_idx] + new_method.split('\n') + lines[insert_idx:]
    content = '\n'.join(new_lines)
    print(f"Step 1 OK: Added _extract_and_upload_images at line {insert_idx}")

# Add image extraction call in process_module
old = '                ModuleService._update_status(conn, module_id, "summarizing")'
new_block = '''                extracted_images = []
                if suffix == ".pdf":
                    try:
                        extracted_images = ModuleService._extract_and_upload_images(tmp_path, module_id, teacher_id)
                        if extracted_images:
                            import json as _json
                            cur.execute("UPDATE modules SET images=%s WHERE id=%s", (_json.dumps(extracted_images), module_id))
                    except Exception as img_err:
                        logger.warning(f"Image extraction failed: {img_err}")

                ModuleService._update_status(conn, module_id, "summarizing")'''

if old in content:
    content = content.replace(old, new_block, 1)
    print("Step 2 OK: Added image extraction call in process_module")
else:
    print("Step 2 SKIP: Could not find summarizing line")

# Update get_module to include images
old_select = 'is_scanned, summary, error_message, token_count, created_at'
new_select = 'is_scanned, summary, error_message, token_count, created_at, images'
content = content.replace(old_select, new_select, 1)

old_return = '"created_at": r[11].isoformat() if r[11] else None,'
new_return = '''"created_at": r[11].isoformat() if r[11] else None,
                "images": json.loads(r[12]) if r[12] and isinstance(r[12], str) else (r[12] or []),'''
if old_return in content:
    content = content.replace(old_return, new_return, 1)
    print("Step 3 OK: Updated get_module to return images")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("\nDONE! Now run:")
print("  copy test-generator-backend\\app\\services\\module_service.py module_service.py")
print("  python api_server.py")
