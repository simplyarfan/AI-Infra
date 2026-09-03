# Hardware appendix

Specifications for every device used in the benchmarking, kept in one place so
the results table stays readable. Numbers are from Apple and NVIDIA's own
published specifications unless noted otherwise.

## MacBook Pro

- Model: MacBook Pro 14 inch, November 2024
- Chip: Apple M4
- Memory: 16 GB unified memory

This is the same 16 GB figure referenced throughout the repo when talking about
what model sizes are realistic to run locally. The MLX benchmark script also
reports peak memory used per run, in the results csv, which shows how much of
that 16 GB a given model actually consumes.

## iPhone 17 Pro

- Chip: Apple A19 Pro, 3nm process
- CPU: 6 core, 2 performance cores at 4.26 GHz, 4 efficiency cores
- GPU: 6 core Apple GPU with Neural Accelerators in each core
- Neural Engine: 16 core
- RAM: 12 GB LPDDR5X
- Cooling: new laser welded vapor chamber, using a small amount of deionised
  water sealed inside the aluminium unibody frame to move heat away from the
  chip. This is new to this generation. Apple's own stated claim is that it
  gives up to 40 percent better sustained performance than the previous
  generation. This is directly relevant to the thermal results in this repo,
  since it is a specific engineering change aimed at exactly the problem being
  measured.
- Released: September 2025

Source: Apple's iPhone 17 Pro technical specifications and product pages.

## iPad Pro 11 inch (4th generation)

Confirmed from the device's own settings info: 11 inch iPad Pro, 4th generation.

- Chip: Apple A12Z Bionic
- CPU and GPU: 8 core CPU, 8 core GPU
- RAM: 6 GB LPDDR4X
- Cooling: passive only, no fan, larger aluminium body than a phone giving more
  thermal mass to absorb heat before it needs to be shed
- Released: March 2020

Source: Apple and Wikipedia's iPad Pro (4th generation) specification pages.

Note: this is a five to six year old chip by the time of testing, considerably
older than the iPhone 17 Pro's A19 Pro. Any comparison between the two should
account for the generation gap, not just phone versus tablet form factor.

## iPhone 15

Not yet tested, planned for tonight.

- Chip: Apple A16 Bionic
- RAM: 6 GB
- Cooling: no vapor chamber, standard design

Alongside the 17 Pro, this gives a two point generational comparison and helps
isolate how much of the 17 Pro's milder throttling comes from its new vapor
chamber cooling versus just being a newer, more efficient chip generally.

## Windows PC

- GPU: NVIDIA RTX 5060 Ti, 16 GB VRAM
- CPU, system RAM, and cooling setup: not yet filled in here, need to confirm
  and add.

This is the most important device on the list for closing a gap in the repo,
since it is the only one with an NVIDIA GPU, meaning it is the only device that
can actually run vLLM and SGLang directly rather than only demonstrating their
underlying principles on other hardware.

## Devices still to add if available

Any additional iPhone or Android models, and ideally a second desktop or laptop
GPU for comparison, would strengthen the generational and cross platform
picture further.
