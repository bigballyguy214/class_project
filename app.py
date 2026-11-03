import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from db import ROOT, connect, execute


def search(query):
    q = query.get('q', [''])[0].strip()[:120]
    postcode = query.get('postcode', [''])[0].strip()[:12]
    amenity = query.get('amenity', [''])[0].strip()[:40]
    clauses, params = [], []
    if q:
        # Escape LIKE wildcards so user input is a literal substring.
        escaped = q.lower().replace('!', '!!').replace('%', '!%').replace('_', '!_')
        clauses.append("LOWER(name) LIKE %s ESCAPE '!'")
        params.append('%' + escaped + '%')
    for column, value in [('postcode', postcode), ('amenity', amenity)]:
        if value:
            clauses.append(column + ' = %s')
            params.append(value)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    conn = connect()
    try:
        # Counts use all matches, independently of the display limit.
        with conn:
            summary = dict(execute(conn,
                "SELECT COUNT(*) AS venue_count, COUNT(NULLIF(opening_hours, '')) AS with_hours, "
                "COALESCE(SUM(CASE WHEN outdoor_seating = 'yes' THEN 1 ELSE 0 END), 0) AS outdoor_count "
                'FROM venues' + where, params).fetchone())
            rows = [dict(r) for r in execute(conn,
                'SELECT osm_id,name,amenity,addr,postcode,opening_hours,outdoor_seating FROM venues'
                + where + ' ORDER BY name, osm_id LIMIT 100', params).fetchall()]
        return {'summary': summary, 'venues': rows, 'display_limit': 100}
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path)
        if path.path == '/':
            return self.send(200, (ROOT / 'templates/index.html').read_bytes(), 'text/html; charset=utf-8')
        if path.path not in ('/api/venues', '/health'):
            return self.send(404, b'{"error":"Not found"}', 'application/json')
        try:
            if path.path == '/health':
                conn = connect()
                try:
                    execute(conn, 'SELECT COUNT(*) FROM venues').fetchone()
                finally:
                    conn.close()
                result = {'status': 'ok'}
            else:
                result = search(parse_qs(path.query))
            self.send(200, json.dumps(result).encode(), 'application/json')
        except Exception:
            # Never return a connection string or database exception to visitors.
            logging.error('Database request failed; check configuration and initialization.')
            self.send(503, b'{"error":"Database unavailable. Check setup and seed status."}', 'application/json')

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8000'))), Handler).serve_forever()
