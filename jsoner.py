import json

js={'credit_cost': 1,
 'credits_monthly_total': 2000,
 'credits_monthly_used': 43,
 'data_type': 'alpr_results',
 'epoch_time': 1521140533533.0,
 'img_height': 384,
 'img_width': 457,
 'processing_time': {'plates': 66.9554672241211,
                     'total': 69.00299999983872,
                     'vehicles': None},
 'regions_of_interest': [{'height': 384, 'width': 457, 'x': 0, 'y': 0}],
 'results': [{'candidates': [{'confidence': 68.77668762207031,
                              'matches_template': 1,
                              'plate': 'BAU7690'}],
              'confidence': 68.77668762207031,
              'coordinates': [{'x': 251, 'y': 340},
                              {'x': 347, 'y': 339},
                              {'x': 346, 'y': 376},
                              {'x': 252, 'y': 377}],
              'matches_template': 1,
              'plate': 'BAU7690',
              'processing_time_ms': 36.96599578857422,
              'region': 'tx',
              'region_confidence': 95.0,
              'requested_topn': 1,
              'vehicle': None,
              'vehicle_region': {'height': 275,
                                 'width': 275,
                                 'x': 147,
                                 'y': 109}}],
 'version': 2}

print(js['results'][0]['plate'])
print('X = ',js['results'][0]['coordinates'][0]['x'],'Y = ',js['results'][0]['coordinates'][0]['y'])
print('X = ',js['results'][0]['coordinates'][2]['x'],'Y = ',js['results'][0]['coordinates'][2]['y'])
