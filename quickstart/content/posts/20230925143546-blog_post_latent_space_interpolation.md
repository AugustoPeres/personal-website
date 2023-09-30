+++
title = "Latent Space Differentiation"
author = ["Augusto"]
description = """
  The MNIST dataset has been used extensively in machine learning for
  benchmarking, demonstration and educational purposes. It seems that all
  introductory machine learning books start with an example on MNIST
  classification and all image generation tutorials start with a MNIST example.

  Here we further the amount of MNIST tutorials but focus instead on a different
  concept: _differentiating with respect to a neural network's inputs_. We will
  see:

  -   How we can combine a classifier and an unconditional autoencoder to generate

  the digits that we want;

  -   How this relates to early text-to-image-models;
  -   Other possible applications.
  """
draft = false
tags = ["python", "machine learning"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++

{{< figure src="/ox-hugo/output.gif" >}}


## Introduction {#introduction}

The MNIST dataset has been used extensively in machine learning for
benchmarking, demonstration and educational purposes. It seems that all
introductory machine learning books start with an example on MNIST
classification and all image generation tutorials start with a MNIST example.

Here we further the amount of MNIST tutorials but focus instead on a different
concept: _differentiating with respect to a neural network's inputs_. We will
see:

-   How we can combine a classifier and an unconditional autoencoder to generate

the digits that we want;

-   How this relates to early text-to-image-models;
-   Other possible applications.

Full code availability [here](https://github.com/AugustoPeres/MNIST-latent-differentiation).


## The classifier {#the-classifier}

The first ingredient in this tutorial is an MNIST classifier. Because there are
hundreds of tutorials on this we will not go into detail here.

Training was done using a simple [pytorch lightning](https://lightning.ai/) wrapper:

```python
class MNISTClassifier(pl.LightningModule):

    def __init__(self, n_layers, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        model = torch.nn.Sequential(torch.nn.Flatten(),
                                    torch.nn.Linear(28 * 28, 128))
        for _ in range(n_layers):
            model.append(torch.nn.Linear(128, 128))
            model.append(torch.nn.ReLU())
        model.append(torch.nn.Linear(128, 10))

        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, _):
        loss, _ = self.compute_loss(batch)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        val_loss, val_accuracy = self.compute_loss(batch)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss, 'val_accuracy': val_accuracy}

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        accuracy_fn = Accuracy(task='multiclass', num_classes=10)
        accuracy = accuracy_fn(logits, y)
        return loss, accuracy

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```

Training is then very simple to implement using lightning. The basic skeleton is
presented below:

```python
# ... train_classifier.py ... #
# ... snip ... #
mnist_classifier = MNISTClassifier(n_layers)
trainer = pl.Trainer()
trainer.fit(
    mnist_classifier,
    torch.utils.data.DataLoader(train_dataset,
                                batch_size=batch_size),
    torch.utils.data.DataLoader(validation_dataset,
                                batch_size=batch_size))
```

The full training script can be found [here](https://github.com/AugustoPeres/MNIST-latent-differentiation/blob/main/train_classifier.py).


## The autoencoder {#the-autoencoder}

The second part in generating numbers using a classifier and an autoencoder is,
off-course, to train an autoencoder. In this section we will briefly introduce
autoencoders and some terminology necessary for the next sections.


### Definitions {#definitions}

An autoencoder in basically an neural network that maps the inputs to the
outputs, _i.e_, an identity function. Basically we want to train a neural
network to learn the function:

<div class="latex">

\begin{align}
f(x) = x, x \in X
\end{align}

</div>

Where, in our particular, case \\(X\\) is the set of MNIST images.

Having a neural netowrk learn the identity function is a trivial task, we could
just, at every layer copy the inputs to the next layer. Therefore, autoencoders
would not be of any particular interest were it not for what is know as a
_"bottleneck layer"_. A _"bottleneck layer"_ is simply a layer that maps the
inputs to a lower dimension space.

To this lower dimension space we call _latent space_. Therefore, it is natural
that we divide the full autoencoder in two parts. The first, to which we call
the _encoder_, maps the inputs to the latent space. The second, to which we call
the _decoder_, maps the latent space back to the original inputs.

{{< figure src="/ox-hugo/autoencoder.png" caption="<span class=\"figure-number\">Figure 1: </span>Autoencoder scheme. \\(E\\) is the encoder, \\(z\\) is the latent space and \\(D\\) is the deocer." >}}


### Our architecture {#our-architecture}

This being an MNIST problem there is no need to over-complicate with convolution
layers, res-nets and other techniques. Therefore, we implemented a simple
multi-layer-perceptron (MLP) as our autoencoder. In pytorch lightning it looks
something like this:

```python
class EncoderDecoderWrapper(pl.LightningModule):

    def __init__(self, latend_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 784),
            torch.nn.Sigmoid(),
        )
```

In this particular case, for visualization purposes, we used
`latent_dim=2`. Recall that training the autoencoder is simply learning an
identity function, thus, in pytorch lightning, the `training_step` is very
simple to implement:

```python
class EncoderDecoderWrapper(pl.LightningModule):

    def __init__(self, latend_dim, learning_rate=1e-3):
        # ... snip ... #

    def training_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def compute_loss(self, batch):
        x, _ = batch
        z = self.encoder(x.view(x.size(0), -1))
        z = self.decoder(z)
        z = z.view(z.size(0), 1, 28, 28)
        loss = nn.MSELoss()(z, x)
        return loss
```

Look [here](https://github.com/AugustoPeres/MNIST-latent-differentiation/blob/ca6990fd2eb42e2de942281bc765c8bfdcc91647/lightning_module.py#L50) for the full autoencoder module and [here](https://github.com/AugustoPeres/MNIST-latent-differentiation/blob/main/train_autoencoder.py) for the training script.


### Training and visualizing {#training-and-visualizing}

Training this autoencoder on the MNIST dataset is a rather lightweight job and
should run on a laptop even without GPU without any issues. After training the
autoencoder from the last epoch can be accessed in
`autoencoder_logs/lightning_logs`.

You can visualize the latent space of the autoencoder by running the script
[plot_latent_space.py](https://github.com/AugustoPeres/MNIST-latent-differentiation/blob/main/plot_latent_space.py). This will encode all images in the validation dataset to
our 2D latent space and then plot those points color-coded according to the
class. You should obtain something like this:

{{< figure src="/ox-hugo/latent_space.png" caption="<span class=\"figure-number\">Figure 2: </span>Visualization of the images mapped to the latent space." >}}


## Generating numbers on demand {#generating-numbers-on-demand}

We could start to generate numbers with just the autoencoder by simply sampling
random points from the latent space and feeding them to the decoder. However,
this would give us no control over which number is generated. Here we see how to
use the classifier and the decoder to generate a number from a requested class.


### Differentiating with respect to the inputs {#differentiating-with-respect-to-the-inputs}

The generating procedure is actually really simple. Suppose that we want to
generate an image of a number of class \\(c\\).

We freeze the weights of the decoder and the classifier. Then, we sample a point
from the latent, \\(z~Z\\) space and feed it to the decoder, this will generate an
image \\(I\\). If we feed \\(I\\) to the classifier we can obtain the probability of \\(I\\)
being of class \\(c\\), _i.e_, \\(p(I=c)\\).

Because this is end-to-end differentiable we can maximize \\(p(I=c)\\) with respect
to \\(z\\). This can be done by applying several iterations of gradient descent to
\\(z\\). In pseudo-code this would look something like:

```python
c = class_to_generate
z = random_latent_space_point
for _ in num_iterations:
    image = decoder(z)
    prob_c = classifier(image)[c]
    # Compute the gradients with respect to z.
    gradients = compute_gradients(prob_c, z)
    z += gradients*learning_rate
```

Or, follow the scheme below:

{{< figure src="/ox-hugo/generative-procedure.png" caption="<span class=\"figure-number\">Figure 3: </span>Shceme of the optimization procedure to generate images using a classifier and a decoder. The decoder generates an image from a point \\(z\\) in the latent space and we move \\(z\\) in order to oprimize the probabilities given by the classifier." >}}


### Implementation in pytorch lightning {#implementation-in-pytorch-lightning}

Again, pytorch lightning makes implementing the following scheme very simple. We
only need to create a `LightningModule` with a classifier, decoder and freeze
their weights. Then, we create a latent space vector and set that as the
trainable parameters:

```python
class MNISTDifferentiableGenerator(pl.LightningModule):
    """Backpropagates with respect to the inputs to generate an image."""

    def __init__(self,
                 classifier,
                 autoencoder,
                 latent_dim,
                 class_to_generate,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier', 'autoencoder'])

        self.learning_rate = learning_rate
        self.class_to_generate = torch.tensor(class_to_generate)

        self.latent = torch.nn.Parameter(
            torch.normal(torch.tensor([0.] * latent_dim),
                         torch.tensor([10.] * latent_dim)))

        self.classifier = classifier
        self.autoencoder = autoencoder
        self.classifier.freeze()
        self.autoencoder.freeze()
```

As we can see `self.latent` is the only trainable parameter in the
`LightningModule`. Now we only need to define the method `training_step`:

```python
def training_step(self, _, __):
    image = self.autoencoder.decoder(self.latent).view(1, 28, 28)
    classifier_logits = self.classifier(image)
    classifier_probs = torch.nn.functional.log_softmax(classifier_logits,
                                                       dim=-1)
    mask_loss = torch.nn.functional.one_hot(
        self.class_to_generate,
        10).type_as(classifier_logits).to(classifier_probs.device)
    loss = torch.sum(-mask_loss * classifier_probs)
    self.log('loss', loss, on_step=True, prog_bar=True)
    self.latents.append(self.latent.detach().cpu())
    return loss
```

This function computes the loss of the _"model"_ which is the negation of the
probabilities of the generated image being of the requested class. We use the
minus sign because, under the hood, pytorch lightning will try to minimize the
loss but we want to maximize the probability. Additionally, we use `log_softmax`
instead of `softmax` because the gradients of the `softmax` function are small
near 1 and 0.

To train this we just need to create a dummy dataloader and train the model:

```python
# generate.py
# ... snip ... #
generator = MNISTDifferentiableGenerator(classifier,
                                         autoencoder,
                                         2,
                                         FLAGS.class_to_generate,
                                         learning_rate=FLAGS.learning_rate)
loader = torch.utils.data.DataLoader(range(FLAGS.iterations_per_epoch),
                                     batch_size=1)
trainer = pl.Trainer()
trainer.fit(generator, model)
```

The full generation script can be found [here](https://github.com/AugustoPeres/MNIST-latent-differentiation/blob/main/generate.py). Try it for your self, play around
with the learning rate and other parameters and hopefully you can obtain
something like this:

{{< figure src="/ox-hugo/output.gif" caption="<span class=\"figure-number\">Figure 4: </span>Left, a visualization of the latent space of the autoencoder. In black we can see the starting point of \\(z\\) in the generation procedure. Center, the image resulting of feeding \\(z\\) to the trained decoder. Right, the probabilities given by the classifier after feeding it the image generated by the decoder." >}}

In the previous image we can visualize the optimization procedure. We requested
a \\(5\\) and the starting position for \\(z\\) (in black) was amongst a cluster of
$0$s. We can see that, throughout several iterations, \\(z\\) successively moves
closer to points in a region in space for which the decoder will generate
\\(5\\). Finally, in the last iterations we can see that \\(z\\) has traveled to a place
that causes the generated image to be classified as a 5 (the requested class).


## Why this matters {#why-this-matters}

Even though this is a simple example it explores the very important topic of
_differentiating with respect to the inputs of the neural network_. In fact this
is how some text-to-image generators work, _e.g_, [VQGAN-CLIP](https://arxiv.org/abs/2204.08583) which has a lot
more moving parts but essentially: we have a model that generates images
(replaces our encoder) and a model ([CLIP](https://openai.com/research/clip)) that scores _"how well the generated
image matches the text prompt"_ (replaces our classifier). Then, just like here,
we optimize until the generated image matches the text description.

Another very different application is that neural networks are essentially
differentiable physical simulators. That is, given initial inputs, we can train
a network to predict the final state of a physical system. And, because
everything is end-to-end differentiable, we can use gradient descent methods to
try and find optimal initial conditions. More on this on [this lecture by
Yuanming Hu](https://www.youtube.com/watch?v=i2O72iMe9ug).


## Further reading {#further-reading}

1.  [Guide to autoencoders](https://www.v7labs.com/blog/autoencoders-guide). This covers autoencoders and variational autoencoders
    very well.;
2.  [CLIP](https://openai.com/research/clip) OpenAI blog post;
3.  [VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language
    Guidance](https://arxiv.org/abs/2204.08583). Uses a similar technique to generate images from text;
4.  [Deep Learning to Discover Coordinates for Dynamics](https://www.youtube.com/watch?v=KmQkDgu-Qp0);
5.  [What happens when OpenAI's CLIP meets BigGAN?](https://www.youtube.com/watch?v=rR5_emVeyBk). In this video, Yannic
    Kilcher makes a video for a song using his text-to-image model made using the
    concepts explored here.
