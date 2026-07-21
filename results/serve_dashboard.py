#!/usr/bin/env python3
"""
Simple HTTP server for viewing GA runs dashboard with live directory listing.
Run with: python3 serve_dashboard.py
Then open: http://localhost:3001/ga_runs_dashboard_embedded.html

Features:
- Serves GA experiments from results/jsons/ directory
- Enables directory listing for jsons/ (allows live experiment discovery)
- CORS headers for cross-origin requests
"""

import http.server
import socketserver
import os
from pathlib import Path

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve files from current directory
        if self.path == '/':
            self.path = '/ga_runs_dashboard_embedded.html'

        # Handle jsons/ directory listing for live experiment discovery
        if self.path == '/jsons/' or self.path == '/jsons':
            jsons_dir = Path('jsons')
            if jsons_dir.exists():
                # Return HTML directory listing (allows frontend to parse filenames)
                experiments = sorted([f.name for f in jsons_dir.glob('experiment_*.json')], reverse=True)

                # Use standard HTML format with proper <a> tags
                html = '''<!DOCTYPE HTML>
<html>
<head><title>Directory listing for /jsons/</title></head>
<body>
<h1>Directory listing for /jsons/</h1>
<ul>
'''
                for exp in experiments:
                    html += f'<li><a href="{exp}">{exp}</a></li>\n'
                html += '''</ul>
</body>
</html>
'''

                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(html.encode())
                return
            else:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return

        # Ensure correct MIME types
        if self.path.endswith('.json'):
            try:
                with open(self.path.lstrip('/'), 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
        elif self.path.endswith('.html'):
            try:
                with open(self.path.lstrip('/'), 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
        else:
            # For other files, add CORS headers to default handler
            super().do_GET()

    def end_headers(self):
        # Add CORS headers to everything
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        # Show all requests
        super().log_message(format, *args)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    PORT = 3001
    Handler = DashboardHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✅ Dashboard server started!")
        print(f"📊 Open browser at: http://localhost:{PORT}/ga_runs_dashboard_embedded.html")
        print(f"🔗 Direct JSON at: http://localhost:{PORT}/ga_runs.json")
        print(f"⏹️  Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✅ Server stopped.")
