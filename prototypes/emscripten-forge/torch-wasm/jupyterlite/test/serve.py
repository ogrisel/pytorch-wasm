import http.server, socketserver, sys, os
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8123
DIR = sys.argv[1]
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()
    def log_message(self, *a): pass
os.chdir(DIR)
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
    print("serving", DIR, "on", PORT, flush=True)
    httpd.serve_forever()
