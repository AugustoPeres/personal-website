+++
title = "Mini Fold"
author = ["Augusto"]
description = """
  This blog post is inspired by the first alpha-fold. Here we generate synthetic
  data and then use neural networks and a simple differentiable potential
  function to obtain the shape of a given sequence of letters.
  """
draft = false
tags = ["python", "machine learning", "numerical optimization", "differential simulators"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++

{{< figure src="/ox-hugo/folding.gif" >}}


# Introduction

In a very simplified fashion the first alpha-fold worked more or less like this:

1. A neural network was trained to predict the distances between all residues of
   the given protein.
2. From those distances a potential function that received the torsion angles
   between residues and outputed the distances between residues was minimized in
   order to find the final shape of the protein.
   
Here, on a much simplified scale, we try to replicate the same process. First we
generate synthetic shapes for sequences of characters. Then, for those sequences
we learn how to predict the distances between all letters. Finally we optimize
with respect to the positions in the cartesian plane until we have a final shape
that respects the distances predicted previously.

The full code for the project can be found on my [github](https://github.com/AugustoPeres/Mini-fold).

# Data generation

First we need to generate synthetic data. Our process is rather simple:

1. We randomly initialize all weights in an embedding;
2. We randomly initialize a recurrent neural network;
3. We feed a random sequence of characters to the rnn and collect all hidden
   states of the network.
4. We randomly initialize an MLP that maps the hidden states to two dimensional
   points (positions in the cartesian plane). We denote this positions as \\(h_i)\\
5. The position if the first letter is set to zero, and the position of the
   \\(i\\)-th letter \\(p_i\\) is defined as \\(p_i = p_{i-1} + h_i\\).
   

However, if we only do this, we are creating shapes for sequences that depend
only on the previous letter. To also consider interaction between subsequent
letters we actually use two RNNs, one of which consumes the sequence
backwards. How much this influences the positions can be controlled by a
discount factor. In python code this looks something like this:


```python
@torch.no_grad
def make_positions(self, sequence):
       f_hidden_states = self.foward_hidden_states(sequence)
       b_hidden_states = self.backward_hidden_states(sequence)

       hidden_states = [
           h_f + self.back_discount * h_b
           for h_f, h_b in zip(f_hidden_states, b_hidden_states)
       ]

       positions = self.position_mlp(torch.stack(hidden_states))

       first_point = positions[0]
       final_positions = [(first_point - first_point).tolist()]
       point = first_point
       for p in positions[1:]:
           point = point + p
           final_positions.append(point.tolist())
       return final_positions
```

For example, to the sequence `abeaefcgeegaccffddeaggdheabbbaggaffd` corresponds
the shape:

{{< figure src="/ox-hugo/target.png">}}

# Distance predictions

Now we want to reverse this process. That is, for a given sequence, we want to
predict its shape. The first step in our process consists of using a transformer
to create embeddings for each character, then we make pairwise sums of all
embeddings and finally use an MLP to compute the corresponding distances. Or, in
python code:

```python
def forward(self, source, src_key_padding_mask=None):
    source = self.embedding(source)
    source = self.pos_encoder(source)
    output = self.transformer_encoder(
        source, src_key_padding_mask=src_key_padding_mask)
    pairwise_sums = output[:, :, None, :] + output[:, None, :, :]
    distance_predictions = self.distance_mlp(pairwise_sums)
    return distance_predictions
```

Could we not simply use the transformer to directly predict the positions of
each character? Well, yes and no. For a problem as simple as this that might
work. However:

1. This was not the approach that I wanted to replicate.
2. By doing that we lose something very important: _Equivariance_. When shapes
   are rotated, the positions of each letter in the cartesian plane change. As
   such, it will be incredibly challenging for the model to tackle this unless
   we use huge amounts of data augmentation. However, the distances between
   letters do are _invariant_ to rotations as such the model is not affected by
   whichever rotation may have been applied to the data before training.
   
Finally, we simply train this model to minimize the MSE between its distance
prediction and the ground truth.

**Note**: This is not the neural network architecture used in the original paper
but I wanted to iterate fast so I decided to go with something that would work
out-of-the-box and was simple enough to implement.

# Final optimization step

Now that we have a model \\(M\\) that can predict the distances between all
letters we are ready for the final step that will give us the shape of the
sequence. To do this we define a function \\(f(p_1, p_2, ..., p_n)\\) the
computes the distances between all letters. Here \\(p_i\\) denotes the position
of the \\(i\\)-th letter.

As such, finding the final shape for a given sequence \\(s\\) is simply a matter
of optimizing \\(||f(p_1, p_2, ..., p_n) - M(s)||\\) with respect to each
\\(p_i\\).

Or, in python code:

```python
class Folder(l.LightningModule):

    def __init__(self,
                 target_distance_matrix,
                 learning_rate=5e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.length = target_distance_matrix.shape[0]
        positions = [torch.zeros(
            (1, 2))] + [torch.rand(1, 2) for _ in range(self.length - 1)]
        self.positions = nn.Parameter(torch.cat(positions))
        self.target_distance_matrix = target_distance_matrix

    def forward(self):
        return torch.norm(self.positions[:, None] - self.positions[None, :],
                          dim=-1)

    def training_step(self, _, __):
        training_loss = self._compute_loss()

    def _compute_loss(self):
        distances = self.forward()
        return nn.MSELoss()(distances, self.target_distance_matrix)

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.parameters(),
                                 lr=self.learning_rate,
                                 max_iter=1)
```

**Note**: It is very important here to use the `LBFGS` optimizer. I was having a
hard time making this last step work until I got back to the original paper and
noticed that this is the optimizer that they use. As soon as I changed it
everything worked.

Here we have a short animation of this process finding the final shape for the
previously shown sequence:

{{< figure src="/ox-hugo/folding.gif" >}}

# Further reading

1. If you want to dive deeper into the first alpha-fold consider reading the
   [original
   paper](https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf).
2. If you want to learn how to improve on the first alpha-fold read the paper
   for the [second
   alpha-fold](https://www.nature.com/articles/s41586-021-03819-2).
3. More on [differentiable simulators](https://arxiv.org/abs/2407.05560).
4. Play around with my [code](https://github.com/AugustoPeres/Mini-fold). For
   example change the structures to be 3D instead of 2D.
