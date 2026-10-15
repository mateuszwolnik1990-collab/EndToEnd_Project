-- Schemat bazy danych SQLite dla systemu wieloagentowego odkrywania aplikacji

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'IDLE',
    pid INTEGER DEFAULT NULL,
    vllm_calls_count INTEGER NOT NULL DEFAULT 0,
    executive_summary TEXT DEFAULT '',
    final_recommendation TEXT DEFAULT '',
    final_report_md TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    problem_statement TEXT NOT NULL,
    target_audience TEXT NOT NULL,
    value_proposition TEXT NOT NULL,
    score_total INTEGER NOT NULL DEFAULT 0,
    score_breakdown_json TEXT NOT NULL,
    is_top3 INTEGER NOT NULL DEFAULT 0,
    claims_json TEXT NOT NULL,
    competitors_analysis TEXT DEFAULT '',
    pricing_analysis TEXT DEFAULT '',
    build_difficulty TEXT DEFAULT '',
    sales_channels TEXT DEFAULT '',
    recommendation TEXT DEFAULT '',
    approved_for_mvp INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS mvp_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL,
    idea_id INTEGER NOT NULL,
    idea_title TEXT NOT NULL,
    mvp_spec_md TEXT NOT NULL,
    architecture_overview TEXT NOT NULL,
    tech_stack_json TEXT NOT NULL,
    core_features_json TEXT NOT NULL,
    database_schema_sql TEXT NOT NULL,
    api_endpoints_json TEXT NOT NULL,
    generated_files_json TEXT NOT NULL,
    sandbox_dir TEXT NOT NULL,
    qa_test_plan TEXT NOT NULL,
    static_security_audit TEXT NOT NULL,
    safe_run_instructions TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE
);
