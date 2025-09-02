
import kornia as K
import torch
import torchvision

# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super().__init__()
        self.l2_term = torch.nn.MSELoss(reduction="mean")
        self.regularization_term = K.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image


# read the image with kornia and add a random noise to it
def apply_tv_denoising(path, output_path):
    noisy_image = K.io.load_image(path, K.io.ImageLoadType.RGB32)  # CxHxW
    tv_denoiser = TVDenoise(noisy_image)
    # define the optimizer to optimize the parameter of tv_denoiser
    optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)

    num_iters: int = 200
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser().sum()
        if i % 50 == 0:
            print(f"Loss in iteration {i} of {num_iters}: {loss.item():.3f}")
        loss.backward()
        optimizer.step()

    img_clean = tv_denoiser.get_clean_image()
    torchvision.utils.save_image(img_clean, output_path)

