+++
title = "Taming transformers for image synthesis implementation"
author = ["Augusto"]
description = """A simple implementation of the paper: Taming transformers for
image synthesis."""
draft = false
tags = ["python", "machine learning", "pytorch", "transformers", "auto-encoders", "generative AI"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++

# Introduction

This post contains a simple pytorch implementation of the paper: [Taming
transformers for high-resolution image
synthesis](https://arxiv.org/abs/2012.09841).

You can check the complete code [on my
github](https://github.com/AugustoPeres/Transformers-for-image-synthesis) where
you will also find instructions on how to install everything and run the code
yourself.

# Method overview

Transformers are currently the state of the art for sequence
generation. However, the architecture itself does not lend itself well to image
generation tasks. Even though images can be represented as their RGB sequences
having a transformer generating such sequence would be computationally very
inefficient. Suppose that we wanted to generate a 64 by 64 image. The
transformer would have to generate \\(64 * 64 * 3 = 12288\\) tokens in order for
us to obtain a very low resolution image.

What the authors in the [Taming transformers for high-resolution image
synthesis](https://arxiv.org/abs/2012.09841) paper did instead was, train an
auto-encoder with a discrete latent space, called a
[VQVAE](https://arxiv.org/abs/1711.00937) and then have the transformer model
learn the distribution of the latent space of the trained VQVAE. The transformer
could then generate novel sequences with the same distribution of the latent
space of the VQVAE which could be given to the decoder to obtain a new image.

# VQVAE

Vector Quantized Variational auto-encoders are a special type of auto-encoders
that have a discrete latent space. That is, every point in the latent space is
represented by using just vectors of some finite collection, called the
**codebook**, \\(E = \\{e_1, e_2, ..., e_k\\}\\) where \\(k\\) is the number of
codebook vectors.

The architecture of VQVAEs consists of and encoder \\(z_e\\), a quantizer layer
\\(z_q\\) and a decoder \\(z_d\\). The encoder and decoder layers work as
expected. The quantizer layer replaces each vector in \\(z_e(x)\\) by the vector
\\(e_i\\) in the codebook with minimal distance to it. For example, suppose that
we have an encoder mapping images with dimension `(32, 32, 1)` images to latent
vectors `z` with shape `(7, 7, 16)`, then our quantizer layer will replace every
vector `z[i, j]` for `i, j in [(0, 0), (1, 0), ..., (6, 6)]` with the
corresponding vector in \\(E\\) that has minimal distance to it.

This means that, once we have a trained VQVAE, we can map our training images to
sequences of size `7 * 7 = 49` corresponding to the codebook indices. These
sequences can then modeled with a transformer.

{{< figure src="/ox-hugo/vqvae-scheme.png" caption="<span class=\"figure-number\">Figure 1: </span>Visualization of the VQVAE architecture. Credits to the original paper." >}}

# Transformer

The transformer part of the image generation process is a standard sequence
modeling task and therefore will not be described here. The only thing that
changes is the masking of the tokens. To both help the transformer scale and
help it understand which token to attend to, a special attention mask is
used. This is called sliding window attention and the transformer is only
allowed to attend to \\(w\\) tokens above and bellow the current token. This is
shown in the picture below:

{{< figure src="/ox-hugo/sliding-window.png" caption="<span class=\"figure-number\">Figure 2: </span>Visualization of the sliding window attention mechanism. Credits to the original paper" >}}

# Results

We implemented the concepts in this paper and trained on the MNIST
dataset. Samples below:


Feel free to try this out with your own dataset.

{{< figure src="/ox-hugo/taming-transformers-samples.png" caption="<span class=\"figure-number\">Figure 2: </span>Visualization of the sliding window attention mechanism." >}}

# Further reading

* [Taming transformers for high-resolution image synthesis
paper](https://arxiv.org/abs/2012.09841) paper;
* [VQVAE paper](https://arxiv.org/abs/1711.00937);
* [Yannic Kilcher's](https://www.youtube.com/watch?v=j4xgkjWlfL4) video on the
  DALLE models.


