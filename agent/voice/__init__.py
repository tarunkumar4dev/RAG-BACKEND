"""a4ai Agent voice layer: speech -> transcript -> the SAME agent loop -> events -> one spoken sentence.

New files only (agent/voice/*), mounted from agent/api.py. Raw audio is never stored or logged; transcript text is
only ever the normal chat message of the session (same history, retention and clear-all as typed chat).
"""
