# Implicit neural differentiable FM synthesizer

[![Replicate](https://replicate.com/andreasjansson/fmsynth/badge)](https://replicate.ai/andreasjansson/fmsynth)

The purpose of this project is to emulate arbitrary sounds with FM synthesis, where the parameters of the FM synth are learned by optimization.

This idea was conceived and implemented during the [Neural Audio Synthesis Hackathon 2021](https://signas-qmul.github.io/nash/). Thanks to Ben Hayes for organizing the workshop and to Mia Chiquier for pointing me towards SIREN!

## Architecture

Please refer to `FMNet` and `Envelope` in `synth.py` for the actual architectural details.

This model takes as input a list of time steps `t_1, t_2, ...`, sampled at some sample rate, and outputs an audio signal in the same sample rate.

Similar to [SIREN](https://arxiv.org/abs/2006.09661), it feeds the input time step values through sinusoidal activation functions initialized with specific weights. In this work we initialize weights to 127 musical pitches from C#-1 to G9. We call this layer the "carrier".

We only use a single sinusoidal layer, but we modulate the frequencies of this layer with a summed output from a separate cosine layer with 127 cosine nodes, also initialized from musical pitches C#-1 to G9. We refer to this layer as the "modulator"

Each carrier and modulator node has both a frequency and an amplitude component. We learn a global phase in the range `(0, 2*pi)` that is shared among all carrier and modulator frequencies. This is effectively a global "bias" term to the sinusoidal activation functions.

The goal of this project is to provide a differentiable emulation of a simple FM synthesizer, so we take a softmax of both carrier and modulator layers' amplitudes.

In addition to carrier and modulator amplitudes we also learn separate amplitude envelope curves for each carrier and modulator node. The envelope is modeled by the bell curve function `1 / sqrt((1 + t * slope) + (slope + offset))`.

## Optimization

This model learns a implicit neural representation for a target audio signal. This means that we optimize the network once for every target signal.

We use the L2 loss between the generated audio signal and the target audio signal as the main loss function.

We also provide optional additional loss terms that maximize the "spikiness" of carrier and modulator amplitude vectors, in order to make the network pick a single carrier and modulator frequency. This term is optional since it sometimes learns more interesting sounds when several carrier and modulators are active.

We use the ADAM optimizer with a learning rate of 0.01.

## Inference

Since this is an implicit neural representation, we can generate outputs at arbitrary sample rates and resolutions. This allows for seamless time stretching and upscaling.

The inference code also supports "stereo detuning" to create musically interesting sounds.
