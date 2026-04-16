CREATE TABLE IF NOT EXISTS subjects (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    grade_level INTEGER NOT NULL CHECK (grade_level BETWEEN 1 AND 12),
    UNIQUE(subject_id, name, grade_level)
);

CREATE TABLE IF NOT EXISTS exercises (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    context TEXT,
    instructions TEXT,
    source_page TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    exercise_id INTEGER REFERENCES exercises(id) ON DELETE CASCADE,
    difficulty INTEGER DEFAULT 3 CHECK (difficulty BETWEEN 1 AND 5),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    explanation TEXT,
    choices JSONB,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Primary analytics grain: one row per student doing one exercise
CREATE TABLE exercise_attempts (
    id BIGSERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    exercise_id INTEGER NOT NULL REFERENCES exercises(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER NOT NULL,
    score_pct NUMERIC(5,2) GENERATED ALWAYS AS (
        CASE WHEN total_questions = 0 THEN 0
             ELSE (correct_answers::numeric * 100.0 / total_questions)
        END
    ) STORED,
    time_seconds INTEGER,
    needs_help BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Optional diagnostics: question-by-question outcome within one exercise attempt
CREATE TABLE exercise_attempt_items (
    id BIGSERIAL PRIMARY KEY,
    exercise_attempt_id BIGINT NOT NULL REFERENCES exercise_attempts(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    is_correct BOOLEAN NOT NULL,
    response_text TEXT,
    response_seconds INTEGER,
    sort_order INTEGER NOT NULL
);

CREATE VIEW student_topic_progress AS
SELECT
  ea.student_id,
  ea.topic_id,
  COUNT(*) AS attempts,
  AVG(ea.score_pct) AS avg_score_pct,
  AVG(ea.time_seconds) AS avg_time_seconds,
  MAX(ea.created_at) AS last_attempt_at
FROM exercise_attempts ea
GROUP BY ea.student_id, ea.topic_id;

CREATE VIEW student_exercise_progress AS
SELECT
  ea.student_id,
  ea.exercise_id,
  COUNT(*) AS attempts,
  MAX(ea.score_pct) AS best_score_pct,
  AVG(ea.score_pct) AS avg_score_pct,
  MAX(ea.created_at) AS last_attempt_at
FROM exercise_attempts ea
GROUP BY ea.student_id, ea.exercise_id;

CREATE INDEX IF NOT EXISTS idx_exercises_topic ON exercises(topic_id);
CREATE INDEX IF NOT EXISTS idx_topics_subject_grade ON topics(subject_id, grade_level);
CREATE INDEX IF NOT EXISTS idx_questions_exercise ON questions(exercise_id);
CREATE INDEX IF NOT EXISTS idx_attempts_question ON attempts(question_id);
CREATE INDEX idx_ex_attempt_items_attempt ON exercise_attempt_items(exercise_attempt_id, sort_order);
CREATE INDEX idx_ex_attempts_student_topic ON exercise_attempts(student_id, topic_id, created_at DESC);
CREATE INDEX idx_ex_attempts_student_exercise ON exercise_attempts(student_id, exercise_id, created_at DESC);
