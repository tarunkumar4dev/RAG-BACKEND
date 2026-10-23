| Method | Route | Limit / min | Counted per |
|---|---|---|---|
| POST | `/api/v1/test-generator/generate-frontend` | 20 | user if logged in, else IP |
| POST | `/api/v1/test-generator/export` | 30 | user if logged in, else IP |
| POST | `/api/v1/test-generator/export-answer-key` | 30 | user if logged in, else IP |
| GET | `/api/v1/test-generator/templates` | 1200 | user if logged in, else IP |
| GET | `/api/v1/test-generator/chapters` | 1200 | user if logged in, else IP |
| POST | `/api/v1/test-generator/generate` | 20 | user if logged in, else IP |
| POST | `/api/v1/test-generator/verify-answers` | 20 | user if logged in, else IP |
| POST | `/api/v1/test-generator/feedback` | 30 | user if logged in, else IP |
| POST | `/api/v1/test-generator/save` | 60 | user if logged in, else IP |
| POST | `/api/v1/test-generator/tests/{test_id}/add-manual-question` | 60 | user if logged in, else IP |
| POST | `/api/v1/test-generator/quiz/create` | 60 | user if logged in, else IP |
| GET | `/api/v1/test-generator/test/{test_id}` | 60 | user if logged in, else IP |
| GET | `/api/v1/test-generator/ncert-questions` | 1200 | user if logged in, else IP |
| GET | `/api/v1/test-generator/ncert-question-stats` | 1200 | user if logged in, else IP |
| POST | `/api/v1/contests` | 60 | user if logged in, else IP |
| GET | `/api/v1/contests/{short_code}/info` | 1200 | user if logged in, else IP |
| POST | `/api/v1/contests/{short_code}/start` | 1200 | user if logged in, else IP |
| POST | `/api/v1/contests/{contest_id}/submit` | 30 per attempt / 1200 per IP | user if logged in, else X-Attempt-Id (all attempts from one IP ≤ 3000), else IP |
| GET | `/api/v1/contests/{contest_id}/leaderboard` | 1200 | user if logged in, else IP |
| GET | `/api/v1/contests/my` | 60 | user if logged in, else IP |
| GET | `/api/v1/payment/plans` | 60 | user if logged in, else IP |
| GET | `/api/v1/payment/plan-status/{user_id}` | 60 | user if logged in, else IP |
| POST | `/api/v1/payment/quote` | 60 | user if logged in, else IP |
| POST | `/api/v1/payment/create-order` | 10 | user if logged in, else IP |
| POST | `/api/v1/payment/verify-payment` | 20 | user if logged in, else IP |
| POST | `/api/v1/payment/webhook` | exempt | — |
| POST | `/api/v1/payment/check-usage/{user_id}` | 60 | user if logged in, else IP |
| POST | `/api/v1/community-quizzes/preview-video` | 20 | user if logged in, else IP |
| POST | `/api/v1/community-quizzes` | 20 | user if logged in, else IP |
| GET | `/api/v1/community-quizzes` | 60 | user if logged in, else IP |
| GET | `/api/v1/community-quizzes/q/{slug}` | 1200 | user if logged in, else IP |
| POST | `/api/v1/community-quizzes/q/{slug}/start` | 1200 | user if logged in, else IP |
| POST | `/api/v1/community-quizzes/q/{slug}/submit` | 30 per attempt / 1200 per IP | user if logged in, else X-Attempt-Id (all attempts from one IP ≤ 3000), else IP |
| GET | `/api/v1/community-quizzes/{quiz_id}/leaderboard` | 60 | user if logged in, else IP |
| GET | `/api/v1/community-quizzes/q/{slug}/leaderboard` | 1200 | user if logged in, else IP |
| DELETE | `/api/v1/community-quizzes/{quiz_id}` | 60 | user if logged in, else IP |
| POST | `/api/v1/modules/create` | 60 | user if logged in, else IP |
| POST | `/api/v1/modules/process` | 10 | user if logged in, else IP |
| GET | `/api/v1/modules/list` | 60 | user if logged in, else IP |
| GET | `/api/v1/modules/{module_id}` | 60 | user if logged in, else IP |
| DELETE | `/api/v1/modules/{module_id}` | 60 | user if logged in, else IP |
| POST | `/api/v1/modules/{module_id}/generate-test` | 20 | user if logged in, else IP |
| POST | `/api/v1/modules/{module_id}/generate-worksheet` | 20 | user if logged in, else IP |
| POST | `/api/v1/modules/{module_id}/download-worksheet` | 60 | user if logged in, else IP |
| POST | `/api/v1/worksheet/generate-direct` | 20 | user if logged in, else IP |
| POST | `/api/v1/worksheet/download` | 60 | user if logged in, else IP |
| POST | `/modules/create` | 60 | user if logged in, else IP |
| POST | `/modules/process` | 10 | user if logged in, else IP |
| GET | `/modules/list` | 60 | user if logged in, else IP |
| GET | `/modules/{module_id}` | 60 | user if logged in, else IP |
| DELETE | `/modules/{module_id}` | 60 | user if logged in, else IP |
| POST | `/modules/{module_id}/generate-test` | 20 | user if logged in, else IP |
| POST | `/modules/{module_id}/generate-worksheet` | 20 | user if logged in, else IP |
| POST | `/modules/{module_id}/download-worksheet` | 60 | user if logged in, else IP |
| POST | `/worksheet/generate-direct` | 20 | user if logged in, else IP |
| POST | `/worksheet/download` | 60 | user if logged in, else IP |
| POST | `/api/v1/test-checker/extract-key` | 10 | user if logged in, else IP |
| POST | `/api/v1/test-checker/check` | 20 | user if logged in, else IP |
| POST | `/api/v1/test-checker/check-with-test` | 20 | user if logged in, else IP |
| GET | `/api/v1/test-checker/results/{check_id}` | 60 | user if logged in, else IP |
| GET | `/api/v1/test-checker/history` | 60 | user if logged in, else IP |
| POST | `/api/v1/chat` | 20 | user if logged in, else IP |
| GET | `/api/v1/whatsapp/webhook` | exempt | — |
| POST | `/api/v1/whatsapp/webhook` | exempt | — |
| POST | `/api/v1/whatsapp/generate_worker` | exempt | — |
| POST | `/api/v1/smart-assignment-checker/check` | 20 | user if logged in, else IP |
| GET | `/api/v1/smart-assignment-checker/result/{check_id}` | 60 | user if logged in, else IP |
| GET | `/api/v1/smart-assignment-checker/history` | 60 | user if logged in, else IP |
| POST | `/api/v1/smart-assignment-checker/assignments/{assignment_id}/rubric` | 10 | user if logged in, else IP |
| POST | `/api/v1/smart-assignment-checker/assignments/{assignment_id}/submissions/{submission_id}/check` | 60 | user if logged in, else IP |
| POST | `/api/v1/copy-checker/sessions/{session_id}/rubric` | 10 | user if logged in, else IP |
| POST | `/api/v1/copy-checker/sheets/{sheet_id}/check` | 60 | user if logged in, else IP |
| PATCH | `/api/v1/copy-checker/sheets/{sheet_id}/questions/{q_no}/marks` | 120 | user if logged in, else IP |
| GET | `/health` | exempt | — |
| GET | `/` | exempt | — |
| GET | `/r/{token}` (G6, not built yet) | 30 | IP |

Default for everything not listed: 60/min — production has no RATE_LIMIT_PER_MINUTE set, so this is the code default. Unmatched paths (404) share one bucket per client.
