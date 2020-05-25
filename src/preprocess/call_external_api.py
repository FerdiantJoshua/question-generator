import http.client
import json


API_BASE_URL = '10.181.131.244'
POS_TAG_PORT = '5500'
NER_PORT = '10009'

def get_pos_tag(text):
    conn = http.client.HTTPConnection(API_BASE_URL, POS_TAG_PORT)
    payload = json.dumps({'text': text})
    headers = {
        "content-type": "application/json",
        "x-api-key": ""
        }

    conn.request('POST', '/', payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode('utf-8'))

    return data

def get_ner(text):
    conn = http.client.HTTPConnection(API_BASE_URL, NER_PORT)
    payload = json.dumps({'text': text})
    headers = {
        "content-type": "application/json",
        "x-api-key": ""
        }

    conn.request('POST', '/', payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode('utf-8'))

    return data
