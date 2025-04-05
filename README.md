# Audio_deepfakes

So 3 models which will be great for our Audio deepfake detection are:

1. Res2net:
*This uses residual networks which should allow it to build deeper models with similar number of layers.
*This uses processing of Audio using Mel-spectrograms, which should improve its accuracy where feature analysis is important.
*As I saw into it more, it is widely used in Audio detection tasks.

2. RawNet2
* This directly uses the audio signals which will save the work of prepressing of signals.
* This way, you will also be able to perform Audio deepfake tasks, where the voices of different person as it can see various patterns from the voice directly.

3. LCNN
* This is a CNN based model which is very simple to build.
* It is very efficient and you can generally train models with high parameters with lower layers and hardware usage.


Because I wanted to do feature extraction and I have experience with CNN, I will train a model based on CNN using the dataset.
In this I have taken just the top 1000 files of the dataset and divided it into training and testing to be able to train for more epochs.
So I achieved about 75% accuracy.
