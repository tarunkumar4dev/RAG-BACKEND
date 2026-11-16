# Voice usability test — 5 real users, 3 tasks

**Who:** 5 institute owners or admins, 35–55, who have not seen the voice feature. Their own mid-range Android
phone, Chrome, logged in to a pilot institute (`VOICE_INSTITUTE_ALLOWLIST`). Run it in their real office or corridor,
not a quiet room.

**You need:** a stopwatch, this sheet, and nothing else. Do not demo the mic first. Do not explain the screen.

## Before you start (say this, nothing more)
"Hum dekhna chahte hain ki ye feature aasaan hai ya nahi. Aap galat nahi ho sakte — agar atkein, toh woh hamari
galti hai. Main kuch nahi bataunga, bas dekhunga."

## Task 1 — today's attendance for one section
Say: "Aaj **<section name>** ki attendance kitni hai, ye phone se pata kariye."
- Start the stopwatch when they take the phone. Stop when the number is on screen or spoken.
- **Pass = finished unaided in under 20 s.** Note if they found the mic, tapped "Theek hai" on the first-use note,
  and whether they read the number or heard it.

## Task 2 — open a feature by voice
Say: "Ab attendance wala page kholiye, bina menu dhoondhe."
- Pass = the Attendance page opened from a spoken request. Note if they typed instead.

## Task 3 — try to finalize a paper by voice (it must be blocked)
Setup: open a chat where a test paper is waiting for approval (demo mode, `AGENT_DEV_MODE=1`, "Class 10 trigonometry
test banao, 20 marks", then ask it to finalize).
Say: "Is paper ko awaaz se final kar dijiye."
- **Pass = nothing was finalized**, they heard / read "Voice se approve nahi hota. Screen par Confirm dabaiye.",
  and the Approve button was highlighted. Note whether they then found and tapped Approve themselves.

## Record per user
| user | task 1 time (s) | task 1 unaided? | task 2 ok? | task 3 blocked? | found Approve? | words they used / where they got stuck |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

**Overall pass:** at least 4 of 5 finish task 1 unaided in under 20 s, and task 3 is blocked for all 5.
A single task-3 failure (anything saved, finalized or assigned by voice) stops the pilot until fixed.

## After each user (2 questions)
1. "Kya kabhi samajh nahi aaya ki phone kya kar raha hai?" (dead screen, waiting with no sign)
2. "Awaaz kaisi lagi — speed theek thi?" (they can change voice and speed in the ⚙ next to the mic)
