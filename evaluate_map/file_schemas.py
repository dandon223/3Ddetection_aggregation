"""Json schemas for files"""
sequences = {
    'type': 'array',
    'minProperties': 1,
    'items': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'sample_token': {
                    'type': 'string'
                },
                'boxes': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'center': {
                                'type': 'array',
                                'items': {
                                    'type': 'number'
                                },
                                'minItems': 3,
                                'maxItems': 3
                            },
                            'size': {
                                'type': 'array',
                                'items': {
                                    'type': 'number'
                                },
                                'minItems': 3,
                                'maxItems': 3
                            },
                            'orientation': {
                                'type': 'array',
                                'items': {
                                    'type': 'number'
                                },
                                'minItems': 4,
                                'maxItems': 4
                            },
                            'name': {
                                'type': 'string'
                            },
                            'score': {
                                'type': 'number'
                            },
                            'label': {
                                'type': [
                                    'null',
                                    'integer'
                                ]
                            },
                            'attribute_name': {
                                'type': 'string'
                            },
                            'num_pts': {
                                'type': 'number'
                            },
                            'velocity': {
                                'type': 'array',
                                'items': {
                                    'type': 'number'
                                },
                                'minItems': 2,
                                'maxItems': 2
                            }
                        },
                        'required': [
                            'center',
                            'size',
                            'orientation',
                            'name',
                            'score'
                        ],
                        'additionalProperties': False
                    }
                }
            },
            'required': [
                'sample_token',
                'boxes'
            ],
            'additionalProperties': False
        }
    }
}
ground_truth_schema = {
    'type': 'object',
    'properties': {
        'sequences': sequences
    },
    'required': [
        'sequences'
    ],
    'additionalProperties': False
    }
generated_annotations_schema = {
    'type': 'object',
    'properties': {
        'sequences': sequences
    },
    'required': [
        'sequences'
    ],
    'additionalProperties': False
    }
