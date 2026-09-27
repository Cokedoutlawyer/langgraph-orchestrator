import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from hybrid_control.adapters import SystemOneProvider
from hybrid_control.core import ContractError


class AdapterTest(unittest.TestCase):
    def test_http_native_receipt_and_provider_specific_confidence(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({'answers':{'eligibility':{'type':'choice','choice':'POSITIVE',
                                  'confidence':.75,'answer_confidence':.96}}}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def log_message(self,*args): pass
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        provider = SystemOneProvider('laya',f'http://127.0.0.1:{server.server_port}/v1/systemone')
        result = provider.evaluate({'text':'evidence'})
        self.assertEqual(result['outcome'],'POSITIVE')
        self.assertEqual(result['confidence'],.96)
        self.assertEqual(result['native']['answers']['eligibility']['confidence'],.75)
        self.assertEqual(set(received[0]['questions']['eligibility']['criteria']),
                         {'POSITIVE','NEGATIVE','UNKNOWN','NOT_APPLICABLE'})
        with self.assertRaises(ContractError):
            SystemOneProvider('jev','http://example.com/v1/systemone')
