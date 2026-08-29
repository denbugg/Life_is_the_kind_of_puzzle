# V32 task

Improve the 24x24 puzzle assembly pipeline by training on newly generated
clean/noisy tile pairs and by replacing aggregate candidate selection with a
slightly larger spatial board critic.  Every source image is 480x480, split
into 576 tiles of 20x20, shuffled with random filenames, and corrupted
independently per tile.  The primary unknown is whether corruption-consistent
features plus a 24x24 error-map critic can close the existing candidate-oracle
gap without overfitting the development scenes.

Run mode: interactive autonomous experiments on the existing RTX 4060 server.
All work remains isolated under `.weco/puzzle-global-autoresearch-v32-noise/`
until a candidate passes the fixed-scene promotion gate.
