import logging
import socket # Import for more specific exception handling
import threading
from pythonosc import dispatcher, osc_server, udp_client

logger = logging.getLogger(__name__)

class OSCManager:
    """
    Manages OSC communication for sending breath data and receiving M4L status.
    """
    def __init__(self, send_ip: str, send_port: int, receive_ip: str, receive_port: int):
        """
        Initializes the OSC client and server.

        Args:
            send_ip: IP address to send OSC messages to (M4L plugin).
            send_port: Port to send OSC messages to.
            receive_ip: IP address to listen for incoming OSC messages on.
            receive_port: Port to listen for incoming OSC messages on.
        """
        try:
            self.client = udp_client.SimpleUDPClient(send_ip, send_port)
            logger.info(f"OSC client configured to send to {send_ip}:{send_port}")
        except Exception as e:
            logger.error(f"Failed to initialize OSC client: {e}")
            self.client = None

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/plugin/status/connected", self._handle_connection_status)
        # Add more mappings here if M4L needs to send other messages

        self.server_address_str = f"{receive_ip}:{receive_port}" # Store for logging
        try:
            self.server = osc_server.ThreadingOSCUDPServer(
                (receive_ip, receive_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self._run_server, name="OSCServerThread")
            self.server_thread.daemon = True  # Daemonize thread
            logger.info(f"OSC server configured to listen on {self.server_address_str}")
        except (socket.error, OSError) as e: # Catch socket-specific errors
            logger.error(f"Failed to initialize OSC server on {self.server_address_str}: {e}")
            self.server = None
            self.server_thread = None

        self.is_m4l_connected = False

    def _run_server(self):
        if self.server:
            try:
                logger.info(f"OSC Server starting on {self.server_address_str}")
                self.server.serve_forever()
            except Exception as e:
                logger.error(f"OSC server error: {e}")
            finally:
                logger.info("OSC server has shut down.")

    def start_server(self):
        """Starts the OSC server in a separate thread."""
        if self.server_thread and not self.server_thread.is_alive():
            self.server_thread.start()
        elif not self.server_thread:
            logger.warning("OSC server was not initialized. Cannot start.")


    def stop_server(self):
        """Stops the OSC server if it is running."""
        if self.server:
            logger.info("Attempting to shut down OSC server...")
            self.server.shutdown() # Signal server_forever to stop
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5) # Wait for thread to finish
                if self.server_thread.is_alive():
                    logger.warning("OSC server thread did not terminate gracefully.")
            logger.info("OSC server stopped.")
        self.server = None # Release server resources


    def _handle_connection_status(self, address: str, *args):
        """
        Handles the /plugin/status/connected OSC message from M4L.
        Expects a single boolean or integer argument (True/1 for connected, False/0 for disconnected).
        """
        if args and isinstance(args[0], (bool, int)):
            self.is_m4l_connected = bool(args[0])
            status_str = "Connected" if self.is_m4l_connected else "Disconnected"
            logger.info(f"OSC: M4L Connection Status updated to: {status_str} (received: {args[0]} from {address})")
        else:
            logger.warning(f"OSC: Received malformed or no argument for M4L connection status from {address}: {args}")

    def get_m4l_connection_status(self) -> bool:
        """Returns the current M4L connection status."""
        return self.is_m4l_connected

    def send_message(self, address: str, value):
        """Sends an OSC message if the client is available."""
        if self.client:
            try:
                self.client.send_message(address, value)
                # logger.debug(f"OSC sent: {address} {value}") # Optional: for verbose logging
            except Exception as e:
                logger.error(f"Failed to send OSC message {address} {value}: {e}")
        else:
            logger.warning(f"OSC client not available. Cannot send message: {address} {value}")


    def send_filtered_differential_signal(self, value: float):
        """Sends the filtered differential signal."""
        self.send_message("/breath/signal/differential", float(value))

    def send_processed_level_signal(self, value: float):
        """Sends the processed level signal."""
        self.send_message("/breath/signal/level", float(value))

    def send_breath_phase(self, phase: str):
        """
        Sends the breath phase (e.g., "inhaling", "exhaling", "neutral").
        """
        self.send_message("/breath/phase", str(phase)) # Ensure it's a string
