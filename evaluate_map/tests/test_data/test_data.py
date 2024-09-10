dt_json = {
    "annotation_source": {
    "model_type": "TED-S",
    "repo_path": "https://ben.ii.pw.edu.pl/gitlab/darts/darts-main/src/services/annotation_generator",
    "checkpoint": "ckpt/nuscenes/TED-S.pth",
    "date_created": "2023-07-01",
    "code_commit": "033faed0c05ebca4328c6ceb79a5bfb57b40d1d3",
    "train_data": [
        "cc8c0bf57f984915a77078b10eb33198",
        "bebf5f5b2a674631ab5c88fd1aa9e87a",
        "2fc3753772e241f2ab2cd16a784cc680",
        "d25718445d89453381c659b9c8734939",
        "de7d80a1f5fb4c3e82ce8a4f213b450a",
        "e233467e827140efa4b42d2b4c435855"
    ]
    },
    "sequences": {
    "cc8c0bf57f984915a77078b10eb33198": [
        {
            "sample_token": "ca9a282c9e77460f8360f564131a8af5",
            "boxes": [
                {
                    "center": [
                        0,
                        0,
                        0.5
                    ],
                    "size": [
                        1,
                        1,
                        1
                    ],
                    "orientation": [
                        1,
                        0,
                        0,
                        0
                    ],
                    "name": "car",
                    "score": 0.8377534598112106,
                },
                {
                    "center": [
                        10,
                        10,
                        10
                    ],
                    "size": [
                        1,
                        1,
                        1
                    ],
                    "orientation": [
                        0.9249,
                        0,
                        0,
                        0.3801
                    ],
                    "name": "pedestrian",
                    "score": 0.8377534598112106,
                }
            ]
        }
    ]
    }
}
gt_json = {
    "sequences": {
        "cc8c0bf57f984915a77078b10eb33198": [
            {
                "sample_token": "ca9a282c9e77460f8360f564131a8af5",
                "boxes": [
                    {
                        "center": [
                            0,
                            0,
                            0
                        ],
                        "size": [
                            1,
                            1,
                            1
                        ],
                        "orientation": [
                            1,
                            0,
                            0,
                            0
                        ],
                        "name": "car",
                        "score": 0,
                    },
                    {
                        "center": [
                            10,
                            10,
                            10
                        ],
                        "size": [
                            1,
                            1,
                            1
                        ],
                        "orientation": [
                            1,
                            0,
                            0,
                            0
                        ],
                        "name": "pedestrian",
                        "score": 0,
                    }
                ]
            }
        ]
    }
}