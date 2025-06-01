# CDE2701A Aspirational Project: Proposal

I would like to train an audio encoder-decoder transformer model that generates an audio file of an improvised saxophone solo given a backing track consisting of bass, piano and drum parts in a typical jazz band setup.


## Objective

To familiarize myself with PyTorch frameworks, understand how transformer architectures work through building and training a model, and to understand how to create high quality datasets for training.

## Motivation

I would like to combine my passion for both jazz music and machine learning in this project, and embark on this journey of exploration and independent learning. I believe this project will allow me to sharpen my research, software development, critical thinking and engineering skills due to how ambitious it is.

## Methodology

### Dataset creation

Having difficulties finding public datasets for something as niche as jazz music, I intend to create my own dataset by following these steps:
1. Downloading audio recordings of jazz music from YouTube
2. Using a stem separation tool (Demucs by Meta AI research) to separate the background audio ("features") and saxophone audio ("target/label").
3. Tokenizing both audios from each example using an open-source pre-trained neural encoder (Encodec by Meta AI research [[Défossez et al., 2022]](https://arxiv.org/pdf/2210.13438)). This converts each audio file into 4 parallel streams of tokens with a sample rate of 50Hz; and each stream has a vocabulary size of 2048. This tokenisation makes the data suitable for training a transformer model.
4. Using RMS energy of audios to determine suitable sections of the audio that can be clipped for training, then clipping the tokenized audio to those sections.
5. Taking inspiration from MusicGen [[Copet et al., 2023]](https://arxiv.org/pdf/2306.05284), introducing a delay codebook interleaving pattern into the streams for each audio for better convergence.

### Model architecture

The model would be a custom encoder-decoder transformer model, as it is solving a sequence-to-sequence problem. The encoder sequence would be the tokenized audio of the backing track, while the sequence that the decoder produces would be the tokenized audio of the saxophone instrumental.

The number of attention heads, number of layers and embedding dimensions for encoder/decoder are all hyperparameters that will have to be experimented with.

The only notable difference in architecture between this and any standard encoder-decoder would be the presence of multiple (4) parallel streams of tokens instead of just 1. Thus, taking inspiration again from MusicGen, each stream will have its own embedding layer. These embeddings will be summed before being passed on to the rest of the network. At the final layer, there will be 4 fully connected layers in parallel to produce the logits for the 4 token streams. 

## Plan

### Phase 1: Proof of Concept
- Create a small-scale dataset by:
  - Downloading a small number of music files
  - Using stem separation to separate instrument parts from the music files
  - Tokenize audio parts using a pre-trained audio compresssion model (Encodec)
  - Extract suitable sections of the audio to use as training examples
- Develop a small-scale version of the model using the small set of data. Ensure that the model can converge to a reasonable loss by tweaking hyperparameters.

### Phase 2: Large-scale dataset creation
- Create a large-scale dataset by repeating the above steps with more audio files, and adding random perturbations in pitch and tempo.

### Phase 3: Model training
- Likely the most time-consuming phase. Train models with large dataset, experiment with hyperparameters to minimize training losss. Evaluate performance manually (by ear) or with evaluation metrics.
- Very likely to need hardware accelerators to train a model at this scale (estimated number of trainable weights in the order of 10^8)

### Phase 4: Upload to HuggingFace
- Deploy and share the model on HuggingFace

### Phase 5: Possible extensions
The below are tentative goals that may change depending on how fast the previous phases are completed:
- Train the model to be able to execute inference on streamed audio (apply causal masking on encoder, use causal audio tokeniser), and deploy it on an embedded system for real-time streaming inference
- Suggested by Prof. Zhang:
  - Write a paper (with his guidance) on the training process and model architecture experimentation
  - Explore patenting options if results produced are excellent

## Stakeholders

No stakeholders identified as this is an individual project.