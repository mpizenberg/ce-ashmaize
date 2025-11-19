"""
Playwright-based browser session for bypassing Kasada bot detection.
Uses a real Chromium browser to execute JavaScript and handle cookies properly.
Runs in a separate thread to avoid event loop conflicts with Textual.
"""

import logging
import random
import time
import threading
from queue import Queue
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page


class BrowserSession:
    """Manages a Playwright browser session for making API requests in a separate thread."""

    def __init__(self, headless=True):
        self.headless = headless
        self._initialized = False
        self._thread = None
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._stop_event = threading.Event()

    def _browser_thread(self, init_url):
        """Thread function that runs the Playwright browser."""
        try:
            logging.info("Starting Playwright browser in separate thread...")
            with sync_playwright() as playwright:
                # Launch Chromium with realistic settings
                browser = playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                    ]
                )

                # Create context with realistic viewport and user agent
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-US',
                    timezone_id='America/New_York',
                )

                # Additional stealth settings
                context.add_init_script("""
                    // Override navigator.webdriver
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false,
                    });

                    // Add chrome runtime
                    window.chrome = {
                        runtime: {},
                    };

                    // Override permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                """)

                page = context.new_page()

                # Visit the init URL to establish session cookies
                logging.info(f"Initializing session by visiting {init_url}")
                page.goto(init_url, wait_until='networkidle', timeout=30000)

                # Add random delay to appear more human
                time.sleep(random.uniform(1.0, 3.0))

                logging.info("Browser session initialized successfully")
                self._initialized = True

                # Process requests in a loop
                while not self._stop_event.is_set():
                    try:
                        # Check for new requests (with timeout to allow checking stop_event)
                        try:
                            request = self._request_queue.get(timeout=0.5)
                        except:
                            continue

                        request_type = request['type']
                        url = request['url']
                        timeout = request.get('timeout', 30000)

                        # Add human-like delay before request
                        time.sleep(random.uniform(0.5, 2.0))

                        response_data = None
                        error = None

                        try:
                            if request_type == 'get':
                                logging.info(f"Browser GET: {url}")
                                response = page.goto(url, wait_until='networkidle', timeout=timeout)

                                if response is None:
                                    error = "Navigation failed - no response received"
                                else:
                                    # Get response body
                                    body = response.body().decode('utf-8')

                                    # Try to parse as JSON
                                    json_data = None
                                    try:
                                        import json
                                        json_data = json.loads(body)
                                    except:
                                        pass

                                    response_data = {
                                        'status': response.status,
                                        'headers': response.headers,
                                        'body': body,
                                        'json': json_data,
                                        'ok': response.ok,
                                    }

                            elif request_type == 'post':
                                logging.info(f"Browser POST: {url}")
                                data = request.get('data')

                                # Use fetch API to make POST request
                                result = page.evaluate("""
                                    async (params) => {
                                        const response = await fetch(params.url, {
                                            method: 'POST',
                                            headers: {
                                                'Content-Type': 'application/json',
                                            },
                                            body: params.data ? JSON.stringify(params.data) : undefined,
                                        });

                                        const text = await response.text();
                                        let json = null;
                                        try {
                                            json = JSON.parse(text);
                                        } catch (e) {
                                            // Not JSON
                                        }

                                        // Convert headers to object
                                        const headers = {};
                                        response.headers.forEach((value, key) => {
                                            headers[key] = value;
                                        });

                                        return {
                                            status: response.status,
                                            headers: headers,
                                            body: text,
                                            json: json,
                                            ok: response.ok,
                                        };
                                    }
                                """, {'url': url, 'data': data})

                                response_data = result

                        except Exception as e:
                            error = str(e)
                            logging.error(f"Browser request failed: {e}")

                        # Send response back
                        self._response_queue.put({
                            'data': response_data,
                            'error': error,
                        })

                    except Exception as e:
                        logging.error(f"Error in browser thread: {e}")

                # Cleanup
                page.close()
                context.close()
                browser.close()
                logging.info("Browser thread stopped")

        except Exception as e:
            logging.error(f"Failed to start browser thread: {e}")
            self._initialized = False

    def initialize(self, init_url):
        """Initialize the browser in a separate thread."""
        if self._initialized or self._thread is not None:
            logging.warning("Browser session already initialized")
            return

        # Start browser thread
        self._thread = threading.Thread(
            target=self._browser_thread,
            args=(init_url,),
            daemon=True
        )
        self._thread.start()

        # Wait for initialization (with timeout)
        timeout = 30
        start_time = time.time()
        while not self._initialized and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self._initialized:
            raise RuntimeError("Browser initialization timed out")

    def _make_request(self, request_type, url, timeout=30000, data=None):
        """Make a request to the browser thread."""
        if not self._initialized:
            raise RuntimeError("Browser session not initialized. Call initialize() first.")

        # Send request
        request = {
            'type': request_type,
            'url': url,
            'timeout': timeout,
            'data': data,
        }
        self._request_queue.put(request)

        # Wait for response (with timeout)
        try:
            response = self._response_queue.get(timeout=timeout / 1000 + 5)
        except:
            raise RuntimeError("Request timed out waiting for browser response")

        if response['error']:
            raise Exception(response['error'])

        return response['data']

    def get(self, url, timeout=30000):
        """
        Make a GET request using the browser.

        Args:
            url: The URL to request
            timeout: Request timeout in milliseconds

        Returns:
            dict with 'status', 'headers', 'body', 'json' keys
        """
        return self._make_request('get', url, timeout)

    def post(self, url, data=None, timeout=30000):
        """
        Make a POST request using the browser.

        Args:
            url: The URL to request
            data: Optional data to send (will be JSON encoded)
            timeout: Request timeout in milliseconds

        Returns:
            dict with 'status', 'headers', 'body', 'json' keys
        """
        return self._make_request('post', url, timeout, data)

    def close(self):
        """Close the browser and cleanup resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._initialized = False
        logging.info("Browser session closed")


# Singleton instance for use across the application
_browser_instance = None


def get_browser_session(headless=True):
    """Get or create the global browser session instance."""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = BrowserSession(headless=headless)
    return _browser_instance


def close_browser_session():
    """Close the global browser session instance."""
    global _browser_instance
    if _browser_instance:
        _browser_instance.close()
        _browser_instance = None
