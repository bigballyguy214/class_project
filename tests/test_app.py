import os
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from app import Handler, search
from seed import seed
from db import connect

class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'DATABASE_MODE':'sqlite', 'SQLITE_PATH':str(Path(self.tmp.name)/'test.db')})
        self.env.start()
        self.old_url = os.environ.pop('DATABASE_URL', None)
        seed()
    def tearDown(self):
        if self.old_url is not None:
            os.environ['DATABASE_URL'] = self.old_url
        self.env.stop()
        self.tmp.cleanup()
    def test_seed_idempotent(self):
        seed()
        self.assertEqual(search({})['summary']['venue_count'],1511)
    def test_filtered_counts(self):
        query={'postcode':['10003'],'amenity':['bar']}
        data=search(query)
        c=connect()
        expected=c.execute("SELECT COUNT(*) FROM venues WHERE postcode='10003' AND amenity='bar'").fetchone()[0]
        c.close()
        self.assertEqual(data['summary']['venue_count'],expected)
        self.assertGreater(expected,0)
        self.assertTrue(all(v['postcode']=='10003' and v['amenity']=='bar' for v in data['venues']))
    def test_input_is_data(self):
        for value in ["' OR 1=1 --", '%', '_']:
            data=search({'q':[value]})
            self.assertTrue(all(value.lower() in v['name'].lower() for v in data['venues']))
        self.assertEqual(search({})['summary']['venue_count'],1511)
    def test_http(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            self.assertIn(b'Nightlife Autopilot',urlopen(base).read())
            self.assertIn(b'"status": "ok"',urlopen(base+'/health').read())
            self.assertIn(b'venue_count',urlopen(base+'/api/venues?q=Stonewall').read())
            with patch('app.connect',side_effect=RuntimeError('private-secret')):
                with self.assertRaises(HTTPError) as e:urlopen(base+'/health')
                self.assertEqual(e.exception.code,503)
                self.assertNotIn(b'private-secret',e.exception.read())
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
