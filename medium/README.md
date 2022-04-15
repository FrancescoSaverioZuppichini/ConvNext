# Implementing ConvNext in PyTorch

Hello There!! Today we are going to implement the famous ConvNext in PyTorch proposed in [A ConvNet for the 2020s
](https://arxiv.org/abs/2201.03545).

Code is [here](https://github.com/FrancescoSaverioZuppichini/ConvNext), an interactive version of this article can be downloaded from [here](https://github.com/FrancescoSaverioZuppichini/ConvNext/blob/main/README.ipynb).

Let's get started!

The paper proposes a new convolution-based architecture that not only surpasses Transformer-based model (such as Swin) but also scales with the amount of data! The following pictures show ConvNext accuracy against the different datasets/models sizes.


<img src="./images/accuracy_table.png" width="500px"></img>


So the authors started by taking the well know ResNet architecture and iteratively improving it following new best practices and discoveries made in the last decade. The authors focused on Swin-Transformer and follows closely its design choices.  The paper is top-notch, I highly recommend read it :) 

The following image shows all the various improvements and the respective performance after each one of them. 

<img src="./images/convnext_improvements.png" width="500px"></img>

They divided their roadmap into two parts: macro design and micro design. Macro design is all the changes we do from a high-level perspective, e.g. the number of stages, while micro design is more about smaller things, e.g. which activation to use.

We will now start with a classic BottleNeck block and apply each change one after the one.

### Starting point: ResNet

As you know (if you don't I have an [article about implementing ResNet in PyTorch](https://medium.com/p/a7da63c7b278)) ResNet uses a residual BottleNeck block, this will be our starting point.


https://gist.github.com/a4760fc0fb8bfab85ed2c298c6b74742

Let's check if it works


https://gist.github.com/dcf783f83605cdcb53c2c85998a27ea5




    torch.Size([1, 64, 7, 7])



Let's also define a `Stage`, a collection of `blocks`. Each stage usually downsamples the input by a factor of `2`, this is done in the first block.


https://gist.github.com/af968df444746bbd2dda8a6da9a259cb


https://gist.github.com/7e4760451158591d82b2bb4aa6aedac9




    torch.Size([1, 64, 4, 4])



Cool, notice how the input was reduced from `7x7` to `4x4`.

ResNet also has what is called `stem`, the first layer in the model that does the heavy downsampling of the input image. 


https://gist.github.com/424f885248f54c602f07671689b3b462

Cool, now we can define `ConvNextEncoder` that holds a list of stages and takes an image as input producing the final embeddings.


https://gist.github.com/417a33e94b4ad671d831c7c957565168


https://gist.github.com/24a2cac86e5e4046d72e6b48ed58e45e




    torch.Size([1, 2048, 7, 7])



This is your normal `resnet50` encoder, if you attach a classification head you get back the good old resnet50 ready to be train on image classification tasks.

## Macro Design

### Changing stage compute ratio

In ResNet we have 4 stages, Swin Transformer uses a ratio of `1:1:3:1` (so one block in the first stage, one in the second, third in the third one ...). Adjusting ResNet50 to this ratio (`(3, 4, 6, 3)` -> `(3, 3, 9, 3)`) results in a performance increase from `78.8%` to `79.4%`.


https://gist.github.com/ecb2dd59b63f8131f914be0d1e908411

### Changing stem to “Patchify”

ResNet stem uses a very aggressive 7x7 conv and a maxpool to heavily downsample the input images. However, Transformers uses a "patchify" stem, meaning they embed the input images in patches. Vision Transfomers uses very aggressive patching (16x16), the authors use 4x4 patch implemented with conv layer. The accuracy changes from `79.4%` to `79.5%` suggesting patching works.


https://gist.github.com/c654f3bc25d6dde1f16b5450312afa4e

### ResNeXt-ify

[ResNetXt](https://arxiv.org/abs/1512.03385) employs grouped convolution for the 3x3 conv layer in the BottleNeck to reduce FLOPS. In ConvNext, they use depth-wise convolution (like in MobileNet and later in EfficientNet). Depth-wise convs are grouped convolutions where the number of groups is equal to the number of input channels. 

The authors notice that is very similar to the weighted sum operation in self-attention, which mixes information only in the spatial dimension. Using depth-wise convs reduce the accuracy (since we are not increasing the widths like in ResNetXt), this is expected. 

So we change our 3x3 conv inside `BottleNeck` block to 

https://gist.github.com/3eb6590d8db385bda8eedc0a9cdfc980

### Inverted Bottleneck

Our BottleNeck first reduces the features via a 1x1 conv, then it applies the heavy 3x3 conv and finally expands the features to the original size. An inverted bottleneck block, does the opposite. I have a [whole article](https://medium.com/p/89d7b7e7c6bc) with nice visualization about them.

So we go from `wide -> narrow -> wide` to `narrow -> wide -> narrow`. 

This is similar to Transformers, since the MLP layer follows the `narrow -> wide -> narrow` design, the second dense layer in the MLP expands the input's feature by a factor of four.


https://gist.github.com/58e3e43216f63d23653b2e7d65c84319

### Large Kernel Sizes

Modern Vision Transfomer, like Swin, uses a bigger kernel size (7x7). Increasing the kernel size will make the computation more expensive, so we move up the big depth-wise conv, by doing so we will have fewer channels. The authors note this is similar to Transformers model where the Multihead Self Attention (MSA) is done before the MLP layers.


https://gist.github.com/c8d5817511b4113b3e77f13a7b1fcfc2

This increases accuracy from `79.9%` to `80.6%`

## Micro Design
### Replacing ReLU with GELU

Since GELU is used by the most advanced transformers, why not use it in our model? The authors report the accuracy stays unchanged. In PyTorch GELU in `nn.GELU`.

### Fewer activation functions

Our block has three activation functions. While, in Transformer block, there is only one activation function, the one inside the MLP block. The authors removed all the activations except for the one after the middle conv layer. This improves accuracy to `81.3%` matching Swin-T!


### Fewer normalization layers

Similar to activations, Transformers blocks have fewer normalization layers. The authors decide the remove all the BatchNorm and kept only the one before the middle conv.

### Substituting BN with LN

Well, they substitute the BatchNorm layers with LinearyNorm. They note that doing so in the original ResNet hurts performance, but after all our changes, the performance increases to `81.5%`

So, let's apply them


https://gist.github.com/2a85412408afa4c86834c3351404979c

### Separate downsampling layers.

In ResNet the downsampling is done by the `stride=2` conv. Transformers (and other conv nets too) have a separate downsampling block. The authors removed the `stride=2` and add a downsampling block before the three convs using a `2x2` `stride=2` conv. Normalization is needed before the downsampling operation to maintain stability during training. We can add this module to our `ConvNexStage`. Finally, we reach `82.0%` surpassing Swin!


https://gist.github.com/3e6b2e6916545bcf8cc10914380006fc

Now we can clean our `BottleNeckBlock`


https://gist.github.com/1c08cafc96b7aa0867d8002aa2e4fd7c

We finally arrive to our final block! Let's test it


https://gist.github.com/c36b7d13d917fc614b4543930d3b9fa8




    torch.Size([1, 62, 7, 7])



## Final touches

They also added Stochastic Depth, also known as Drop Path, (I have an [article](https://towardsdatascience.com/implementing-stochastic-depth-drop-path-in-pytorch-291498c4a974) about it) and Layer Scale.


https://gist.github.com/6cd06c1dbe8792561e3689b208aa4da7

et voilà 🎉 We have arrived to the final ConvNext Block! Let's see if it works!


https://gist.github.com/ffcb186f15f601761f4f9ac18dc3fa72




    torch.Size([1, 62, 7, 7])



Super! We need to create the drop paths probabilities in the encoder


https://gist.github.com/dab8601a0b116c1c33988ba719bdacf8


https://gist.github.com/251a61c4febe395d2b6a02e748e63e70




    torch.Size([1, 2048, 3, 3])



To get the final ConvNext used for Image classification we need to apply a classification head on top of the encoder. We also add a `LayerNorm` before the last linear layer. 


https://gist.github.com/57afd8125677a8fd8b26a162180d41ca


https://gist.github.com/d156228dec74b7361cf36c30744002ce




    torch.Size([1, 1000])



And here you have it!

## Conclusions

In this article we have seen, step by step, all the changes the Authors did to create ConvNext from ResNet. I hope this was useful :)

Thank you for reading it!

Francesco
