# Sketch-R2CNN (copy & modification)

Code for Sketch-R2CNN: An Attentive Network for Vector Sketch Recognition. Lei Li et al. arXiv abs/1811.08170, 2018.

---

Instructions:

- `Docker` is prefered for running the code.
    1. Build the Docker image by running `docker/build.sh`.
    2. Training and testing scripts are in the `scripts` folder.

- Otherwise
    1. Install library dependencies on your host computer according to the `docker/Dockerfile`
    2. Remove the `sudo nvidia-docker run...py35pytorch101` prefix in the `scripts/*.sh` files before running.
