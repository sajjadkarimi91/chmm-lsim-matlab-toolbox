# chmm-lsim-matlab-toolbox

> **This repository is no longer developed. It continues at [chmm-lsim-x](https://github.com/sajjadkarimi91/chmm-lsim-x).**
>
> chmm-lsim-x contains the same MATLAB functions together with a Python package (`chmm_lsim`), the POISE inference recursion, discriminative (MMI) training, GPU support and tests. Please use it for new work and report issues there. This repository is kept so that existing code and papers that cite it keep working; it receives no further updates.

A MATLAB toolbox for HMMs, coupled HMMs (CHMMs) and Latent Structure Influence Models (LSIMs) in multi-channel data processing, modeling and classification.

## Final update

The functions in this repository were brought up to date with chmm-lsim-x one last time:

- `forward_backward_lsim` and `em_lsim` share one forward-backward kernel (`forward_backward_lsim_kernel`) with three backward recursions, selected with `lsim_fb_constants`: **POISE** (the default), **SED** and **fast/SVI**.
  - **The default inference changed from SED to POISE.** To reproduce results obtained with earlier versions, pass `lsim_fb_constants('sed')`, for example `extra.fb_constants = lsim_fb_constants('sed')` for `em_lsim`.
- New functions: `em_lsim_poise`, `forward_backward_lsim_poise`, `forward_backward_lsim_fast`, `forward_backward_lsim_kernel`, `forward_backward_lsim_svi`, `lsim_fb_constants`, `lsim_pack_para`.
- Fixes:
  - emission densities of fully missing (`NaN`) samples in `gmm_pdf` and `gmm_pdf_fast`;
  - two errors that stopped `em_lsim_svi` from running.
- `demo_run_lsim.m` and `demo_run_chmm.m` were replaced by short demos that print their results.

Function signatures are unchanged.

## Usage

```matlab
addpath(genpath('<path to this toolbox>'))
demo_run_lsim      % POISE, SED and fast vs exact inference, then EM
demo_run_chmm      % full CHMM vs LSIM
```
