Config = {
    'lr': 0.0001,
    'weight_decay': 0.0,
    'batch_size': 64,

    'time_type': 'cat', # cat or add
    'dims': [1000],
    'norm': False,
    'emb_size': 10,

    'mean_type': 'x0',
    'steps': 100,
    'noise_schedule': 'linear-var',
    'noise_scale': 0.1,
    'noise_min': 0.0001,
    'noise_max': 0.02,

    'sampling_steps': 0,
    'sampling_noise': False,
    'reweight': True,

}