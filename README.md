# SaxGPT: Audio encoder-decoder transformer

SaxGPT is a multi-stream encoder-decoder transformer, trained with a custom dataset to generate saxohphone solos on jazz backing tracks.

### Dataset creation

Having difficulties finding public datasets for something as niche as jazz music, a custom dataset was created by following these steps:
1. Downloading audio recordings of jazz music from YouTube
2. Using a stem separation tool (Demucs by Meta AI research) to separate the background audio ("features") and saxophone audio ("target/label").
3. Tokenizing both audios from each example using an open-source pre-trained neural encoder (Encodec by Meta AI research [[Défossez et al., 2022]](https://arxiv.org/pdf/2210.13438)). This converts each audio file into 4 parallel streams of tokens with a sample rate of 50Hz; and each stream has a vocabulary size of 2048. This tokenisation makes the data suitable for training a transformer model.
4. Using RMS energy of audios to determine suitable sections of the audio that can be clipped for training, then clipping the tokenized audio to those sections.
5. Taking inspiration from MusicGen [[Copet et al., 2023]](https://arxiv.org/pdf/2306.05284), introducing a delay codebook interleaving pattern into the streams for each audio for better convergence.

### Model architecture

The model is a custom encoder-decoder transformer model with KV caching. The encoder sequence would be the tokenized audio of the backing track, while the sequence that the decoder produces would be the tokenized audio of the saxophone instrumental.

The only notable difference in architecture between this and any standard encoder-decoder would be the presence of multiple (4) parallel streams of tokens instead of just 1. Thus, taking inspiration again from MusicGen, each stream will have its own embedding layer. These embeddings will be summed before being passed on to the rest of the network. At the final layer, there will be 4 fully connected layers in parallel to produce the logits for the 4 token streams.
