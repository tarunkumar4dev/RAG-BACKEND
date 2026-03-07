-- ============================================================
-- A4AI Test Generator — Supabase Migration
-- Run this in Supabase SQL Editor
-- ============================================================

-- 1. Vector search function (uses existing ncert_chunks + pgvector)
CREATE OR REPLACE FUNCTION match_ncert_chunks(
  query_embedding vector(384),
  subject_filter text,
  class_filter text,
  chapter_filter text[],
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id int,
  chapter text,
  subject text,
  class_grade text,
  content text,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    nc.id,
    nc.chapter,
    nc.subject,
    nc.class_grade,
    nc.content,
    1 - (nc.embedding <=> query_embedding) AS similarity
  FROM ncert_chunks nc
  WHERE
    LOWER(nc.subject) LIKE '%' || subject_filter || '%'
    AND nc.class_grade = class_filter
    AND (
      array_length(chapter_filter, 1) IS NULL
      OR nc.chapter = ANY(chapter_filter)
    )
    AND 1 - (nc.embedding <=> query_embedding) > match_threshold
  ORDER BY nc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 2. Tests table
CREATE TABLE IF NOT EXISTS tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  teacher_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  exam_title TEXT NOT NULL,
  board TEXT DEFAULT 'CBSE',
  class_grade TEXT,
  subject TEXT,
  iteration INT DEFAULT 0,
  status TEXT DEFAULT 'draft',  -- draft | saved | exported
  request_payload JSONB,         -- full request stored for regeneration
  total_questions INT,
  total_marks INT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Questions table
CREATE TABLE IF NOT EXISTS questions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_id UUID REFERENCES tests(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  options JSONB,                 -- array for MCQ, null for others
  correct_answer TEXT,
  explanation TEXT,
  marks INT DEFAULT 1,
  difficulty TEXT,               -- easy | medium | hard
  bloom_level TEXT,              -- null if bloom disabled
  chapter TEXT,
  topic TEXT,
  format TEXT,                   -- mcq | short_answer | etc
  validation_status TEXT DEFAULT 'verified',
  validation_notes TEXT,
  position INT,                  -- order in test
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Training data (rejected/edited questions — our moat)
CREATE TABLE IF NOT EXISTS training_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_id UUID REFERENCES tests(id) ON DELETE SET NULL,
  teacher_id UUID,
  question_id UUID,
  question_text TEXT,
  correct_answer TEXT,
  teacher_feedback TEXT,
  action TEXT,                   -- reject | edit
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Quizzes (shareable student links)
CREATE TABLE IF NOT EXISTS quizzes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  test_id UUID REFERENCES tests(id) ON DELETE CASCADE,
  teacher_id UUID REFERENCES auth.users(id),
  duration_minutes INT DEFAULT 60,
  max_marks INT,
  passing_marks INT,
  shuffle_questions BOOLEAN DEFAULT TRUE,
  shuffle_options BOOLEAN DEFAULT TRUE,
  camera_required BOOLEAN DEFAULT FALSE,
  tab_switch_limit INT DEFAULT 3,
  status TEXT DEFAULT 'active',  -- active | closed
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Subscriptions (payment feature gating)
CREATE TABLE IF NOT EXISTS subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  plan_type TEXT DEFAULT 'free',  -- free | teacher_pro | institution
  valid_till TIMESTAMPTZ,
  features_allowed JSONB DEFAULT '{
    "tests_per_day": 1,
    "max_questions": 10,
    "bloom_taxonomy": false,
    "file_upload": false,
    "quiz_sharing": false,
    "blueprint_mode": false
  }',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. RLS Policies
ALTER TABLE tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE quizzes ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Teachers see own tests" ON tests
  FOR ALL USING (auth.uid() = teacher_id);

CREATE POLICY "Questions visible via test" ON questions
  FOR ALL USING (
    EXISTS (SELECT 1 FROM tests WHERE tests.id = questions.test_id AND tests.teacher_id = auth.uid())
  );

CREATE POLICY "Users see own subscription" ON subscriptions
  FOR ALL USING (auth.uid() = user_id);

-- 8. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tests_teacher ON tests(teacher_id);
CREATE INDEX IF NOT EXISTS idx_questions_test ON questions(test_id);
CREATE INDEX IF NOT EXISTS idx_training_teacher ON training_data(teacher_id);
CREATE INDEX IF NOT EXISTS idx_ncert_subject ON ncert_chunks(subject);
CREATE INDEX IF NOT EXISTS idx_ncert_chapter ON ncert_chunks(chapter);