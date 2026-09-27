"""Laya local and Jev remote System One HTTP adapters.

Wire contract source: NandhaKishorM/laya, laya/serve.py and README,
inspected 2026-09-27. Laya uses answer_confidence; Jev uses native confidence.
No credentials, URLs, or responses are synthesized when a provider is unavailable.
"""
import json
from urllib import request, error, parse
from .core import ContractError, OUTCOMES


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ContractError('provider redirects are not permitted')


class SystemOneProvider:
    def __init__(self, name, endpoint, *, api_key=None, timeout=10):
        url = parse.urlsplit(endpoint)
        if name not in ('laya','jev') or url.username or url.password or url.query or url.fragment:
            raise ContractError('invalid provider configuration')
        if url.scheme != 'https' and not (name == 'laya' and url.scheme == 'http' and url.hostname in ('localhost','127.0.0.1','::1')):
            raise ContractError('remote providers require HTTPS')
        if not 0 < timeout <= 30:
            raise ContractError('provider timeout must be bounded at 30 seconds')
        self.name, self.endpoint, self.api_key, self.timeout = name, endpoint, api_key, timeout

    def evaluate(self, context):
        payload = {'state': {'body': json.dumps(context, sort_keys=True)}, 'questions': {
            'eligibility': {'type':'choice','instructions':'Classify the supplied evidence. Missing evidence is UNKNOWN.',
                'criteria': {'POSITIVE':'Evidence explicitly establishes the criterion.',
                             'NEGATIVE':'Evidence explicitly refutes the criterion.',
                             'UNKNOWN':'Evidence is absent, insufficient, or contradictory.',
                             'NOT_APPLICABLE':'The criterion explicitly does not apply.'}}}}
        headers = {'Content-Type':'application/json','Accept':'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        req = request.Request(self.endpoint,json.dumps(payload).encode(),headers,method='POST')
        try:
            with request.build_opener(NoRedirect).open(req,timeout=self.timeout) as response:
                body = response.read(1_048_577)
                if len(body) > 1_048_576:
                    raise ContractError('provider response exceeds size contract')
                raw = json.loads(body)
            answer = raw['answers']['eligibility']
            outcome = answer['choice']
            confidence = answer['answer_confidence' if self.name == 'laya' else 'confidence']
            if outcome not in OUTCOMES:
                raise ContractError('provider returned an unknown option')
            return {'outcome':outcome,'confidence':confidence,'native':raw,'provider':self.name}
        except (error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            raise ContractError('provider protocol or availability failure') from exc
