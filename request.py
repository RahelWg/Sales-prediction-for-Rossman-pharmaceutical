import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'store_type':2, 'promo':9, 'comp_distance':6})

print(r.json())# -*- coding: utf-8 -*-

