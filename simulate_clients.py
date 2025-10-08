"""
Multi-Client Simulation Script
Simulates multiple clients locally for testing
"""
import subprocess
import time
import signal
import sys
from pathlib import Path
from config import CONFIG

class ClientSimulator:
    """Simulate multiple clients locally"""
    
    def __init__(self, num_clients: int = 10, mode: str = "lwfedssl"):
        self.num_clients = num_clients
        self.mode = mode
        self.processes = []
        self.server_process = None
    
    def start_server(self):
        """Start the server process"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting {self.mode.upper()} Server")
        print(f"{'='*80}\n")
        
        cmd = [
            sys.executable,
            "enhanced_server.py",
            "--mode", self.mode
        ]
        
        # Start server without capturing output so we can see it
        self.server_process = subprocess.Popen(cmd)
        
        # Give server just 3 seconds to bind to port, then start clients
        # (Server will wait for clients to provide initial parameters)
        print("⏳ Giving server 3 seconds to bind to port...")
        time.sleep(3)
        
        # Check if server is still running
        if self.server_process.poll() is not None:
            print("❌ Server failed to start!")
            sys.exit(1)
        
        print("✅ Server process started, ready for clients...\n")
    
    def start_clients(self):
        """Start multiple client processes"""
        print(f"\n{'='*80}")
        print(f"🔄 Starting {self.num_clients} Clients")
        print(f"{'='*80}\n")
        
        # Use localhost instead of 0.0.0.0 for better connection
        server_address = f"localhost:{CONFIG.SERVER_PORT}"
        if self.mode == "baseline":
            server_address = f"localhost:{CONFIG.SERVER_PORT + 1}"
        
        # Start clients in smaller batches to avoid overwhelming the server
        batch_size = 3
        for batch_start in range(0, self.num_clients, batch_size):
            batch_end = min(batch_start + batch_size, self.num_clients)
            print(f"\n📦 Starting batch {batch_start//batch_size + 1} (clients {batch_start}-{batch_end-1})...")
            
            for i in range(batch_start, batch_end):
                cmd = [
                    sys.executable,
                    "enhanced_client.py",
                    "--server", server_address,
                    "--client-id", str(i),
                    "--mode", self.mode
                ]
                
                print(f"  • Client {i}...", end=" ")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.processes.append(process)
                print("✓")
                time.sleep(1)  # Longer delay between clients
            
            # Wait between batches
            if batch_end < self.num_clients:
                print(f"  ⏳ Waiting 3 seconds before next batch...")
                time.sleep(3)
        
        print(f"\n✅ All {self.num_clients} clients started\n")
    
    def wait_for_completion(self):
        """Wait for all processes to complete"""
        print(f"\n{'='*80}")
        print("⏳ Waiting for training to complete...")
        print(f"{'='*80}\n")
        
        try:
            # Wait for clients
            for i, process in enumerate(self.processes):
                process.wait()
                print(f"✅ Client {i} completed")
            
            # Stop server
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait()
                print("✅ Server stopped")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user, cleaning up...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🧹 Cleaning up processes...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
        
        print("✅ Cleanup complete\n")
    
    def run(self):
        """Run the simulation"""
        try:
            self.start_server()
            self.start_clients()
            self.wait_for_completion()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Client Simulator for LW-FedSSL")
    parser.add_argument("--num-clients", type=int, default=10,
                       help="Number of clients to simulate")
    parser.add_argument("--mode", type=str, default="lwfedssl",
                       choices=["lwfedssl", "baseline"],
                       help="Training mode")
    
    args = parser.parse_args()
    
    simulator = ClientSimulator(num_clients=args.num_clients, mode=args.mode)
    simulator.run()

if __name__ == "__main__":
    main()
