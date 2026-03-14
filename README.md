# General Relativistic Ray Tracing

GPU Accelerated General Relativistic ray tracing around a binary black hole system with blackbody thin accretion disks and doppler brightening.

[![Watch the render](bh_gr_rt.gif)](https://github.com/Kushaalkumar-pothula/gr-ray-tracing/raw/main/binary_bh_rt.mp4)

## Features
- Superposed Schwarzschild geodesic integration (RK4)
- Adaptive step size for accurate disk crossing detection
- Doppler brightening and gravitational redshift
- ACES tone mapping and gamma correction
- ~15s render at 960×540, SPP=4 on an NVIDIA T4


## Roadmap
- [ ] Develop in C++/CUDA
- [ ] Inspiral animation
- [ ] Full radiative transfer (probably)
