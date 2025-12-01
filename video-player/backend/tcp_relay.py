import socket
import threading
import time
import select
import logging
import traceback

logger = logging.getLogger(__name__)

class TCPRelay(threading.Thread):
    """
    Relays video packets from a local internal UDP port to multiple external TCP clients.
    Allows clients to simply connect via TCP to receive the stream.
    """
    
    def __init__(self, internal_port: int, external_port: int):
        super().__init__()
        self.internal_port = internal_port
        self.external_port = external_port
        self.running = False
        self.clients = [] # List of client sockets
        self.lock = threading.Lock()
        self.daemon = True
        
    def run(self):
        self.running = True
        logger.info(f"TCP Relay starting: Internal UDP :{self.internal_port} -> External TCP :{self.external_port}")
        
        sock_in = None
        sock_server = None
        
        try:
            # Socket for receiving video from FFmpeg (Internal UDP)
            sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock_in.bind(('127.0.0.1', self.internal_port))
            sock_in.setblocking(False)
            
            # Socket for accepting clients (External TCP)
            sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock_server.bind(('0.0.0.0', self.external_port))
            sock_server.listen(5)
            sock_server.setblocking(False)
            
            logger.info(f"TCP Relay listening on 0.0.0.0:{self.external_port}")
            
            while self.running:
                # Build list of sockets to read from (UDP In + TCP Server)
                read_list = [sock_in, sock_server]
                
                # Wait for activity
                readable, _, _ = select.select(read_list, [], [], 1.0)
                
                for s in readable:
                    if s is sock_in:
                        # Receive video data from FFmpeg
                        try:
                            data = s.recv(65536)
                            if data:
                                self._broadcast_to_clients(data)
                        except Exception as e:
                            logger.error(f"Error reading from internal UDP: {e}")
                            
                    elif s is sock_server:
                        # Accept new TCP client
                        try:
                            client_sock, addr = sock_server.accept()
                            # Disable Nagle's algorithm for low latency
                            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            client_sock.setblocking(False)
                            
                            with self.lock:
                                self.clients.append(client_sock)
                            
                            logger.info(f"New TCP client connected: {addr}")
                        except Exception as e:
                            logger.error(f"Error accepting TCP client: {e}")
                            
        except Exception as e:
            error_msg = f"TCP Relay crashed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            with open("/tmp/relay_crash.log", "a") as f:
                f.write(f"{time.ctime()}: {error_msg}\n")
        finally:
            self._close_all(sock_in, sock_server)
            logger.info("TCP Relay stopped")
            
    def _broadcast_to_clients(self, data):
        with self.lock:
            if not self.clients:
                return
                
            disconnected = []
            for client in self.clients:
                try:
                    # We use sendall for TCP, but since it's non-blocking, 
                    # we might need to handle partial sends or blocking.
                    # For simplicity in this relay, we try to send.
                    # Ideally, we should use select for writing too, but for video relay
                    # dropping slow clients is often better than blocking everyone.
                    # Let's try blocking send with timeout? No, that blocks main loop.
                    # Let's just try send().
                    client.sendall(data)
                except (BrokenPipeError, ConnectionResetError):
                    disconnected.append(client)
                except BlockingIOError:
                    # Client buffer full - drop packet for this client (congestion control)
                    # Or disconnect them? For now, just skip (drop frame)
                    pass
                except Exception as e:
                    logger.debug(f"Error sending to client: {e}")
                    disconnected.append(client)
            
            for client in disconnected:
                self._remove_client(client)

    def _remove_client(self, client):
        try:
            client.close()
        except:
            pass
        if client in self.clients:
            self.clients.remove(client)
            logger.info("TCP client disconnected")

    def _close_all(self, sock_in, sock_server):
        if sock_in:
            try: sock_in.close()
            except: pass
        if sock_server:
            try: sock_server.close()
            except: pass
        
        with self.lock:
            for client in self.clients:
                try: client.close()
                except: pass
            self.clients.clear()

    def stop(self):
        self.running = False
        self.join()
        
    def get_client_count(self):
        with self.lock:
            return len(self.clients)
    
    def is_alive(self):
        return super().is_alive()
