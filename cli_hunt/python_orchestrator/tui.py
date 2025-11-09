import logging
import threading
from collections import OrderedDict
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Log, Static

# --- Custom Messages for thread-safe UI updates ---


class LogMessage(Message):
    """Message to add a line to the TUI log viewer."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()


class ChallengeUpdate(Message):
    """Message to update the status of a single challenge in the TUI table."""

    def __init__(self, address: str, challenge_id: str, status: str) -> None:
        self.address = address
        self.challenge_id = challenge_id
        self.status = status
        super().__init__()


class RefreshTable(Message):
    """Message to signal a full refresh of the TUI table."""

    pass


class StatsUpdate(Message):
    """Message to update wallet statistics display."""

    def __init__(self, stats: dict, total: float) -> None:
        self.stats = stats  # Dict of {address: total_mined}
        self.total = total
        super().__init__()


# --- The Main TUI Application ---


class OrchestratorTUI(App):
    """A Textual TUI for the Midnight Scavenger Hunt orchestrator."""

    TITLE = "Midnight Scavenger Hunt Orchestrator"
    
    def __init__(
        self, db_manager, worker_functions: dict, worker_args: dict, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.db_manager = db_manager
        self.worker_functions = worker_functions
        self.worker_args = worker_args
        self.stop_event = threading.Event()

        # Internal state for the table
        self._addresses = []
        self._challenge_ids = OrderedDict()  # challenge_id -> short_id
        self._wallet_stats = {}  # address -> total_mined
        self._total_mined = 0.0

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with VerticalScroll():
            yield DataTable(id="challenges_table", cursor_type="row")
        with Horizontal():
            yield Log(id="logs", auto_scroll=True, max_lines=1000)
            with Container(id="stats_container"):
                yield Static("💰 Wallet Mining Statistics", id="stats_header")
                yield Static("TOTAL: 0.000000", id="stats_total")
                yield DataTable(id="stats_table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.log_widget = self.query_one(Log)
        self.table = self.query_one(DataTable)
        self.stats_table = self.query_one("#stats_table", DataTable)
        self.stats_total = self.query_one("#stats_total", Static)

        self.log_widget.write_line("TUI mounted. Initializing table...")
        
        # Load existing wallet statistics from database
        self._wallet_stats = self.db_manager.get_all_wallet_statistics()
        self._total_mined = sum(self._wallet_stats.values())
        
        self.refresh_table_structure()
        self.refresh_stats_table()
        self.run_startup_stats_update()
        
        if self._total_mined > 0:
            self.log_widget.write_line(f"💰 Loaded existing stats: {self._total_mined:.6f} total mined")

        self.log_widget.write_line("Starting background worker threads...")
        self.run_fetcher_worker()
        self.run_solver_worker()
        self.run_saver_worker()
        self.run_stats_worker()

    def _get_status_display(self, status: str) -> str:
        """Return a user-friendly (emoji) string for a status."""
        status_map = {
            "available": "⏳ Avail",
            "solving": "⚙️ Solving",
            "solved": "✅ Solved",
            "validated": "🏆 Validated",
            "expired": "❌ Expired",
            "submission_error": "❗️ Error",
        }
        return status_map.get(status, status)

    def refresh_table_structure(self) -> None:
        """
        Initializes or rebuilds the entire table structure (columns and rows).
        It will always display the latest 15 challenges across all addresses.
        """
        self.table.clear(columns=True)

        # --- Get current data from DB ---
        self._addresses = self.db_manager.get_addresses()
        if not self._addresses:
            self.log_widget.write_line("No addresses found. Table is empty.")
            return

        all_challenge_ids = set()
        for addr in self._addresses:
            for c in self.db_manager.get_challenge_queue(addr):
                all_challenge_ids.add(c["challengeId"])

        # Sort challenges and keep only the most recent 15
        sorted_challenges = sorted(list(all_challenge_ids))
        if len(sorted_challenges) > 15:
            sorted_challenges = sorted_challenges[-15:]

        self._challenge_ids = OrderedDict((cid, cid[3:]) for cid in sorted_challenges)

        # --- Add Columns ---
        self.table.add_column("Address", key="address")
        for cid, short_cid in self._challenge_ids.items():
            self.table.add_column(short_cid, key=cid)

        # --- Add Rows ---
        for addr in self._addresses:
            short_addr = f"{addr[:6]}...{addr[-4:]}"
            # Create a row with placeholders. We'll populate it next.
            row_data = [short_addr] + ["-"] * len(self._challenge_ids)
            self.table.add_row(*row_data, key=addr)

        # --- Populate all cells with current status for displayed challenges ---
        for addr in self._addresses:
            # Only consider challenges that are actually displayed in the table
            displayed_challenges = [
                c
                for c in self.db_manager.get_challenge_queue(addr)
                if c["challengeId"] in self._challenge_ids
            ]
            for c in displayed_challenges:
                self.post_message(ChallengeUpdate(addr, c["challengeId"], c["status"]))

    def refresh_stats_table(self) -> None:
        """Initialize or rebuild the stats table with wallet mining statistics."""
        self.stats_table.clear(columns=True)
        
        # Add columns
        self.stats_table.add_column("Address", key="address", width=15)
        self.stats_table.add_column("Total Mined", key="total_mined", width=15)
        
        # Add rows for each wallet
        for addr in self._addresses:
            short_addr = f"{addr[:6]}...{addr[-4:]}"
            total_mined = self._wallet_stats.get(addr, 0.0)
            total_mined_str = f"{total_mined:.6f}" if total_mined > 0 else "0.000000"
            self.stats_table.add_row(short_addr, total_mined_str, key=addr)
        
        # Update total in header
        self.stats_total.update(f"TOTAL: {self._total_mined:.6f}")

    # --- Message Handlers ---

    def on_log_message(self, message: LogMessage) -> None:
        """Display a log message from a worker."""
        now = datetime.now().strftime("%H:%M:%S")
        self.log_widget.write_line(f"[{now}] {message.message}")
        logging.info(message.message)  # Also write to the actual log file

    def on_challenge_update(self, message: ChallengeUpdate) -> None:
        """Update a single cell in the DataTable, if the challenge is currently displayed."""
        if message.challenge_id in self._challenge_ids:
            try:
                display_status = self._get_status_display(message.status)
                self.table.update_cell(
                    message.address, message.challenge_id, display_status
                )
            except KeyError:
                # This should ideally not happen if _challenge_ids is consistent with table columns
                self.log_widget.write_line(
                    f"[Warning] Could not update cell (KeyError) for displayed challenge: "
                    f"{message.address[:6]}/{message.challenge_id}. This indicates a potential sync issue."
                )
        # If the challenge ID is not in _challenge_ids, we simply ignore the update
        # as it's not a challenge we're currently displaying.

    def on_refresh_table(self, message: RefreshTable) -> None:
        """Handle request to perform a full table refresh."""
        self.log_widget.write_line("Refreshing table data...")
        self.refresh_table_structure()

    def on_stats_update(self, message: StatsUpdate) -> None:
        """Update wallet statistics in the stats table and display total."""
        self._wallet_stats = message.stats
        self._total_mined = message.total
        
        # Update the stats table
        for addr, total_mined in message.stats.items():
            if addr in self._addresses:
                total_mined_str = f"{total_mined:.6f}" if total_mined > 0 else "0.000000"
                try:
                    self.stats_table.update_cell(addr, "total_mined", total_mined_str)
                except KeyError:
                    pass  # Address not in current table view
        
        # Update total in header
        self.stats_total.update(f"TOTAL: {self._total_mined:.6f}")
        
        # Log the total
        self.log_widget.write_line(f"💰 Total mined across all wallets: {self._total_mined:.6f}")

    # --- Actions ---

    def action_quit(self) -> None:
        """Action to quit the application, triggered by Ctrl+C."""
        self.log_widget.write_line("Shutdown signal received. Stopping threads...")
        self.stop_event.set()
        # Give workers a moment to notice the event. A proper implementation would join them.
        self.log_widget.write_line("Performing final save...")
        self.db_manager.save_to_disk()
        self.log_widget.write_line("Exiting.")
        self.exit()

    # --- Worker Definitions ---

    @work(name="fetcher", group="workers", thread=True)
    def run_fetcher_worker(self) -> None:
        """Runs the fetcher logic in a background thread."""
        fetcher_func = self.worker_functions["fetcher"]
        fetcher_func(self.db_manager, self.stop_event, self)

    @work(name="solver", group="workers", thread=True)
    def run_solver_worker(self) -> None:
        """Runs the solver logic in a background thread."""
        solver_func = self.worker_functions["solver"]
        solve_interval = self.worker_args["solve_interval"]
        max_solvers = self.worker_args["max_solvers"]
        challenge_selection = self.worker_args["challenge_selection"]
        solver_func(self.db_manager, self.stop_event, solve_interval, self, max_solvers, challenge_selection)

    @work(name="saver", group="workers", thread=True)
    def run_saver_worker(self) -> None:
        """Runs the database saver logic in a background thread."""
        saver_func = self.worker_functions["saver"]
        interval = self.worker_args["save_interval"]
        saver_func(self.db_manager, self.stop_event, interval, self)

    @work(name="stats", group="workers", thread=True)
    def run_stats_worker(self) -> None:
        """Runs the wallet statistics updater logic in a background thread."""
        stats_func = self.worker_functions["stats"]
        interval = self.worker_args.get("stats_interval", 60 * 60 * 24)  # Default 24 hours
        stats_func(self.db_manager, self.stop_event, interval, self)
        
    @work(name="startup_stats", thread=True)
    def run_startup_stats_update(self) -> None:
        """Fetches fresh wallet statistics on startup."""
        from main import fetch_wallet_statistics
        
        addresses = self.db_manager.get_addresses()
        for address in addresses:
            total_mined = fetch_wallet_statistics(address)
            if total_mined is not None:
                self.db_manager.update_wallet_statistics(address, total_mined)
        
        # Get all stats and calculate total
        all_stats = self.db_manager.get_all_wallet_statistics()
        total = sum(all_stats.values())
        
        # Send stats update to TUI
        self.post_message(StatsUpdate(all_stats, total))
        self.post_message(LogMessage(f"✅ Startup statistics update complete"))
