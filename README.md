## 翻译论文 
## Very Deep Convolutional Networks For Large-Scale Image Recognition
### Abstract
本文，我们研究在大规模图像识别任务中，卷积神经网络的深度对准确率的影响。我们用一个由3X3卷积核组成的网络框架来进行实验。结果表明通过把网络从16层加到19层weight layers，准确率会有很大改善。这一发现使得我们在2014年的ImageNet Challenge比赛的定位和分类任务中分别获得了第一名和第二名的成绩。而且我们的研究成果也适用于其他数据集，其模型也是目前表现最好的。我们已经把两个表现最好的卷积神经网络模型公之于众，从而能够促进机器视觉的发展。

###Itroduction
最近，在大规模图像识别和视频识别中，卷积神经网络发挥了重大作用。这主要归功于大的图像数据库，比如ImageNet，以及计算机性能的提升，比如GPU。尤其是，ImageNet Large-Scale Visual Recognition Challenge(ILSVRC)挑战赛对深度视觉识别框架的发展起到了很大的推动作用。从high-dimentional shallow feature encodings到深度卷积神经网络，ILSVRC挑战赛孵化出了好几代大规模图像分类系统(large-scale image classification systems)。

随着卷积神经网络在计算机视觉领域的商业价值越来越高，很多人试图改进最初Krizhevsky et al,(2012)创建的框架,从而获得更高的准确率。例如，在ILSVRC-2013挑战赛中，表现最好的提交在模型的第一层用更小的窗口和更小的步幅。还有人用不同同一图像的不同尺度来训练网络和测试网络(Sermanet et al.,2014;Howard,2014)。本文，我们着重探索卷积神经网络另一个重要的特征---深度。我们保持网络其他超参数不变，不断的增加卷积层数，并且每一层的卷积核都是3X3大小。

结果，我们得到了准确度更好的神经网络框架，不仅在比赛中取得了最好的成绩，而且该网络框架还适用于其他图像数据集，并且表现很好。
