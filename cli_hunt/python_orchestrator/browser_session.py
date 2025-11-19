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
from playwright_stealth import Stealth


# Randomization data for realistic fingerprinting
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
]

VIEWPORTS = [
    {'width': 1920, 'height': 1080},
    {'width': 1920, 'height': 1200},
    {'width': 2560, 'height': 1440},
    {'width': 1680, 'height': 1050},
    {'width': 1440, 'height': 900},
]

TIMEZONES = [
    'America/New_York',
    'America/Chicago',
    'America/Los_Angeles',
    'America/Denver',
    'Europe/London',
]

LOCALES = [
    'en-US',
    'en-GB',
]


class BrowserSession:
    """Manages a Playwright browser session for making API requests in a separate thread."""

    def __init__(self, headless=True, mining_page_url=None):
        self.headless = headless
        self.mining_page_url = mining_page_url or "https://sm.midnight.gd/wizard/mine"
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
                # Randomize fingerprint for each session
                user_agent = random.choice(USER_AGENTS)
                viewport = random.choice(VIEWPORTS)
                timezone = random.choice(TIMEZONES)
                locale = random.choice(LOCALES)

                logging.info(f"Using User-Agent: {user_agent[:50]}...")
                logging.info(f"Using Viewport: {viewport}")

                # Launch Chromium with realistic settings
                # Use new headless mode (headless=new) which is harder to detect
                browser = playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--no-first-run',
                        '--no-default-browser-check',
                        '--disable-infobars',
                        '--window-size={},{}'.format(viewport['width'], viewport['height']),
                    ]
                )

                # Create context with randomized viewport and user agent
                context = browser.new_context(
                    viewport=viewport,
                    user_agent=user_agent,
                    locale=locale,
                    timezone_id=timezone,
                    # Add extra HTTP headers to appear more realistic
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                    }
                )

                # Enhanced stealth settings - comprehensive fingerprint evasion
                context.add_init_script("""
                    // Override navigator.webdriver
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false,
                    });

                    // Add chrome runtime (critical for passing bot detection)
                    window.chrome = {
                        runtime: {},
                        loadTimes: function() {},
                        csi: function() {},
                        app: {},
                    };

                    // Override permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );

                    // Randomize canvas fingerprint
                    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                    HTMLCanvasElement.prototype.toDataURL = function(type) {
                        const result = originalToDataURL.apply(this, arguments);
                        // Add slight noise to canvas fingerprint
                        return result.replace(/.$/, String.fromCharCode(result.charCodeAt(result.length - 1) + Math.floor(Math.random() * 3)));
                    };

                    // Randomize WebGL fingerprint
                    const getParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        if (parameter === 37445) { // UNMASKED_VENDOR_WEBGL
                            return 'Intel Inc.';
                        }
                        if (parameter === 37446) { // UNMASKED_RENDERER_WEBGL
                            return 'Intel Iris OpenGL Engine';
                        }
                        return getParameter.apply(this, arguments);
                    };

                    // Override plugin detection
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [
                            {
                                name: 'Chrome PDF Plugin',
                                filename: 'internal-pdf-viewer',
                                description: 'Portable Document Format',
                            },
                            {
                                name: 'Chrome PDF Viewer',
                                filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                                description: '',
                            },
                            {
                                name: 'Native Client',
                                filename: 'internal-nacl-plugin',
                                description: '',
                            }
                        ],
                    });

                    // Override languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                    });

                    // Override platform (randomize between common platforms)
                    Object.defineProperty(navigator, 'platform', {
                        get: () => 'MacIntel',
                    });

                    // Override hardwareConcurrency (randomize)
                    Object.defineProperty(navigator, 'hardwareConcurrency', {
                        get: () => """ + str(random.choice([4, 8, 12, 16])) + """,
                    });

                    // Override deviceMemory (randomize)
                    Object.defineProperty(navigator, 'deviceMemory', {
                        get: () => """ + str(random.choice([4, 8, 16])) + """,
                    });

                    // Override battery API
                    if (navigator.getBattery) {
                        navigator.getBattery = () => Promise.resolve({
                            charging: true,
                            chargingTime: 0,
                            dischargingTime: Infinity,
                            level: 1,
                        });
                    }

                    // Add realistic timing jitter
                    const originalDateNow = Date.now;
                    Date.now = function() {
                        return originalDateNow() + Math.floor(Math.random() * 5);
                    };

                    // Mask automation in performance.timing
                    Object.defineProperty(window.performance.timing, 'navigationStart', {
                        get: () => Date.now() - Math.floor(Math.random() * 10000 + 5000),
                    });
                """)

                page = context.new_page()

                # Apply playwright-stealth for comprehensive bot detection evasion
                # This automatically patches dozens of detection vectors
                logging.info("Applying playwright-stealth patches...")
                stealth = Stealth()
                stealth.apply_stealth_sync(page)

                # First visit the home page to establish session cookies
                logging.info(f"Initializing session by visiting {init_url}")
                page.goto(init_url, wait_until='networkidle', timeout=30000)

                # Add random delay to appear more human
                initial_delay = random.uniform(2.0, 4.0)
                logging.info(f"Waiting {initial_delay:.2f}s on home page...")
                time.sleep(initial_delay)

                # Navigate to the mining page (where we'll "stay" during operations)
                logging.info(f"Navigating to mining page: {self.mining_page_url}")
                page.goto(self.mining_page_url, wait_until='networkidle', timeout=30000)

                # Add delay and simulate human-like behavior on mining page
                mining_delay = random.uniform(2.0, 5.0)
                logging.info(f"Waiting {mining_delay:.2f}s to simulate reading mining page...")
                time.sleep(mining_delay)

                # Simulate some human-like behavior on the mining page
                try:
                    # Random scroll to appear more human
                    page.evaluate("""
                        () => {
                            window.scrollTo({
                                top: Math.random() * 500,
                                behavior: 'smooth'
                            });
                        }
                    """)
                    time.sleep(random.uniform(0.5, 1.2))

                    # Scroll back to simulate looking around
                    page.evaluate("""
                        () => {
                            window.scrollTo({
                                top: 0,
                                behavior: 'smooth'
                            });
                        }
                    """)
                    time.sleep(random.uniform(0.3, 0.8))
                except:
                    pass  # Ignore errors if page doesn't support scrolling

                logging.info("Browser session initialized successfully")
                self._initialized = True

                # Track last activity time for keepalive
                last_keepalive = time.time()
                keepalive_interval = random.uniform(180, 300)  # 3-5 minutes

                # Process requests in a loop
                while not self._stop_event.is_set():
                    try:
                        # Periodic keepalive: simulate user activity on the mining page
                        current_time = time.time()
                        if current_time - last_keepalive > keepalive_interval:
                            try:
                                logging.info("Performing keepalive interaction on mining page...")
                                # Ensure we're still on the mining page
                                if page.url != self.mining_page_url:
                                    logging.info(f"Page drifted to {page.url}, navigating back to mining page")
                                    page.goto(self.mining_page_url, wait_until='networkidle', timeout=30000)
                                    time.sleep(random.uniform(1.0, 2.0))

                                # Simulate random user activity
                                activity_choice = random.choice(['scroll', 'click', 'hover'])
                                if activity_choice == 'scroll':
                                    # Random scroll
                                    page.evaluate("""
                                        () => {
                                            const scrollAmount = Math.random() * 300 + 100;
                                            window.scrollBy({
                                                top: scrollAmount,
                                                behavior: 'smooth'
                                            });
                                        }
                                    """)
                                    time.sleep(random.uniform(0.5, 1.0))
                                    # Scroll back
                                    page.evaluate("""
                                        () => {
                                            window.scrollTo({
                                                top: 0,
                                                behavior: 'smooth'
                                            });
                                        }
                                    """)
                                elif activity_choice == 'hover':
                                    # Simulate mouse movement
                                    page.mouse.move(
                                        random.randint(100, 800),
                                        random.randint(100, 600)
                                    )

                                last_keepalive = current_time
                                keepalive_interval = random.uniform(180, 300)  # Randomize next interval
                                logging.info(f"Keepalive complete. Next in ~{keepalive_interval/60:.1f} minutes")
                            except Exception as e:
                                logging.warning(f"Keepalive interaction failed: {e}")
                                last_keepalive = current_time  # Reset anyway to avoid rapid retries

                        # Check for new requests (with timeout to allow checking stop_event)
                        try:
                            request = self._request_queue.get(timeout=0.5)
                        except:
                            continue

                        request_type = request['type']
                        url = request['url']
                        timeout = request.get('timeout', 30000)

                        # Add human-like delay before request (more variation)
                        # Use exponential distribution for more realistic timing
                        base_delay = random.uniform(1.0, 3.0)
                        # Occasionally add a longer pause (10% chance)
                        if random.random() < 0.1:
                            base_delay += random.uniform(2.0, 5.0)
                        time.sleep(base_delay)

                        response_data = None
                        error = None

                        try:
                            if request_type == 'get':
                                logging.info(f"Browser GET: {url}")

                                # Make the request using fetch from the mining page context
                                # This ensures proper Referer header and looks like a real user request
                                result = page.evaluate("""
                                    async (params) => {
                                        const response = await fetch(params.url, {
                                            method: 'GET',
                                            headers: {
                                                'Accept': 'application/json, text/plain, */*',
                                                'Accept-Language': 'en-US,en;q=0.9',
                                                'Sec-Fetch-Dest': 'empty',
                                                'Sec-Fetch-Mode': 'cors',
                                                'Sec-Fetch-Site': 'same-origin',
                                            },
                                            credentials: 'include',
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
                                            url: response.url,
                                        };
                                    }
                                """, {'url': url})

                                response_data = result

                            elif request_type == 'post':
                                logging.info(f"Browser POST: {url}")
                                data = request.get('data')

                                # Make the POST request using fetch from the mining page context
                                # This ensures proper Referer header and looks like a real user request
                                result = page.evaluate("""
                                    async (params) => {
                                        const response = await fetch(params.url, {
                                            method: 'POST',
                                            headers: {
                                                'Content-Type': 'application/json',
                                                'Accept': 'application/json, text/plain, */*',
                                                'Accept-Language': 'en-US,en;q=0.9',
                                                'Sec-Fetch-Dest': 'empty',
                                                'Sec-Fetch-Mode': 'cors',
                                                'Sec-Fetch-Site': 'same-origin',
                                            },
                                            body: params.data ? JSON.stringify(params.data) : undefined,
                                            credentials: 'include',
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
                                            url: response.url,
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
