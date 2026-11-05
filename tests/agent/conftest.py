import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agent.schemas import SessionCtx  # noqa: E402

from _agent_ids import ADMIN, DECOY, DECOY_ADMIN, MAIN, SHARMA, VERMA  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    from agent.db import reset_and_seed
    reset_and_seed()


@pytest.fixture
def admin_ctx():
    return SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)


@pytest.fixture
def sharma_ctx():
    return SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)


@pytest.fixture
def verma_ctx():
    return SessionCtx(user_id=VERMA, role="teacher", institute_id=MAIN, teacher_id=VERMA)


@pytest.fixture
def decoy_ctx():
    return SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY)
