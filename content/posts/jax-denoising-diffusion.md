+++
title = "Denoising diffusion models in jax"
author = ["Augusto"]
description = """A simple implementation of denoising diffusion models on jax.
  """
draft = false
tags = ["python", "machine learning", "jax", "diffusion", "generative AI"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++


{{< figure src="/ox-hugo/denoising_diffusion_sample.gif" >}}


# Diffusion in jax

[This repository](https://github.com/AugustoPeres/diffusion) contains
the bare minimum necessary code to train a simple diffusion model
using [jax](https://github.com/google/jax?tab=readme-ov-file) and
[equinox](https://github.com/patrick-kidger/equinox). We focus on
unconditional diffusion models and our implementation follows
[Denoising Diffusion Probabilistic Models
](https://arxiv.org/abs/2006.11239) exactly.


## Running the code

The first thing you need to run this code is obviously the data. For
that just make sure that all your images are in the same
folder. Additionally they should already be cropped to the same
(square) size. Other than that there are no additional restrictions.

Next, create  virtual environment and install the requirements:

```bash
python3 -m venv .en
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
```


You are now ready to run the code:

```bash
python scripts/train.py --path_to_data=<path_to_your_images> \
                        --output_path=<where_to_save_artifact>
```

For the descriptions of the other flags run:

```bash
python scripts/train.py --help
```

## Example samples

{{< figure src="/ox-hugo/denoising_diffusion_sample.gif" >}}
