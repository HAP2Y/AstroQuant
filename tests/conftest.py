"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_path():
    """Return the project root path."""
    return project_root


@pytest.fixture
def sample_date():
    """Return a sample date for testing."""
    return datetime(2024, 6, 15, 12, 0, 0)


@pytest.fixture
def date_range():
    """Return a sample date range for testing."""
    return datetime(2024, 6, 1), datetime(2024, 6, 30)


@pytest.fixture
def sample_ticker():
    """Return a sample ticker for testing."""
    return "AAPL"


@pytest.fixture
def sample_price():
    """Return a sample price for testing."""
    return 150.0
