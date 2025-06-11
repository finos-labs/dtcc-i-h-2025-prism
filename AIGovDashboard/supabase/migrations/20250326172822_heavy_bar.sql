/*
  # Add initial dummy projects

  1. Changes
    - Insert two initial projects:
      - Investment Portfolio Analysis (Generic AI)
      - Test LLM (LLM)

  2. Notes
    - Projects are assigned to a demo user for testing
    - Status and timestamps are set automatically
*/

-- Insert dummy projects with a demo user ID
INSERT INTO projects (name, description, project_type, status, created_at, updated_at, user_id)
VALUES 
  (
    'Investment Portfolio Analysis',
    'AI-powered investment strategy and risk assessment system',
    'generic',
    'Completed',
    now() - interval '1 day',
    now() - interval '8 hours',
    '00000000-0000-0000-0000-000000000000'
  ),
  (
    'Test LLM',
    'LLM Beta Testing',
    'llm',
    'No Report Yet',
    now() - interval '2 days',
    now() - interval '1 day',
    '00000000-0000-0000-0000-000000000000'
  )
ON CONFLICT DO NOTHING;