"""End-to-end tests for the Streamlit web app using Playwright."""

import subprocess
import time
import os
import pytest
from pathlib import Path
from PIL import Image


# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

FIXTURES_DIR = Path(__file__).parent / "fixtures"
APP_PATH = Path(__file__).parent.parent / "app" / "app.py"


@pytest.fixture
def sample_banknote_image():
    """Create a test banknote-like image."""
    img_path = FIXTURES_DIR / "test_banknote.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        # Create a realistic-sized test image
        img = Image.new("RGB", (600, 300), color=(50, 120, 80))
        img.save(img_path, quality=90)

    return str(img_path)


@pytest.fixture(scope="module")
def app_server():
    """Launch Streamlit app server for testing."""
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    proc = subprocess.Popen(
        ["python", "-m", "streamlit", "run", str(APP_PATH),
         "--server.port", "8599", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )

    # Wait for server to start
    time.sleep(8)

    yield "http://localhost:8599"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
class TestAppE2E:
    """End-to-end tests for the Streamlit currency classifier app."""

    def test_app_loads(self, app_server):
        """Test that the app loads and shows the title."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")

            # Check page title content
            content = page.text_content("body")
            assert "Currency Identifier" in content

            browser.close()

    def test_instructions_visible(self, app_server):
        """Test that usage instructions are shown when no image is uploaded."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")

            content = page.text_content("body")
            assert "How to use" in content
            assert "Upload" in content or "upload" in content

            browser.close()

    def test_file_uploader_exists(self, app_server):
        """Test that the file upload widget is present."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")

            # Streamlit file uploader has a specific structure
            uploader = page.query_selector('[data-testid="stFileUploader"]')
            assert uploader is not None, "File uploader should be present"

            browser.close()

    def test_accessibility_heading_exists(self, app_server):
        """Test that proper heading structure exists for screen readers."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")

            # Check for h1 heading
            h1 = page.query_selector("h1")
            assert h1 is not None, "Page should have an h1 heading"
            heading_text = h1.text_content()
            assert "Currency" in heading_text

            browser.close()

    def test_footer_disclaimer(self, app_server):
        """Test that the disclaimer footer is present."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(app_server, wait_until="networkidle")

            content = page.text_content("body")
            assert "educational purposes" in content.lower() or "accessibility" in content.lower()

            browser.close()
