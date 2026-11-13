-- public.tests as defined in RAG-BACKEND/test-generator-backend/supabase_migration.sql (lines 49-63).
-- The Supabase stub pre-creates a minimal public.tests (id, teacher_id), so add the real columns explicitly.
CREATE TABLE IF NOT EXISTS public.tests (id UUID PRIMARY KEY DEFAULT gen_random_uuid());
ALTER TABLE public.tests
  ADD COLUMN IF NOT EXISTS teacher_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS exam_title TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS board TEXT DEFAULT 'CBSE',
  ADD COLUMN IF NOT EXISTS class_grade TEXT,
  ADD COLUMN IF NOT EXISTS subject TEXT,
  ADD COLUMN IF NOT EXISTS iteration INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'draft',
  ADD COLUMN IF NOT EXISTS request_payload JSONB,
  ADD COLUMN IF NOT EXISTS total_questions INT,
  ADD COLUMN IF NOT EXISTS total_marks INT,
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
