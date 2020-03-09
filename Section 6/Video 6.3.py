import torchvision.datasets
import torchvision.transforms
import torch


loader = torch.utils.data.DataLoader(
             torchvision.datasets.UCF101(
                './UCF101/UCF-101',
                './UCF101/ucfTrainTestlist',
                5
            ))

video, audio, label = next(iter(loader))
print('Video size:', video.shape)
print('Audio size:', audio.shape)
print('Label size:', label.shape)