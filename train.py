import argparse
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset.super_resolution_loader import SuperResolutionUrban100Dataset
from src.modules.generator import SuperResolutionTransformer
from src.modules.discriminator import ContentDiscriminator
from src.utils.loss_utils import *
from src.utils.image_utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train-folder', type=str, default='urban100/image_SRF_4', help='train folder')
    parser.add_argument('--epochs', type=int, default=100000, help='epochs count')
    parser.add_argument('--print-result-per-epochs', type=int, default=1000, help='print result to folder frequency')
    parser.add_argument('--out-folder', type=str, default='output', help='output train results')
    opt = parser.parse_args()

    path = os.path.join(os.path.abspath(os.getcwd()), opt.train_folder)
    out_path = os.path.join(os.path.abspath(os.getcwd()), opt.out_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    seed = 42
    lr = opt.lr
    epochs = opt.epochs
    print_result_per_epochs = opt.print_result_per_epochs

    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    dataset = SuperResolutionUrban100Dataset(path, transform)

    train_length = int(len(dataset) * 0.9)
    valid_length = int(len(dataset) * 0.05)
    test_length = len(dataset) - train_length - valid_length

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_length, valid_length, test_length])
    train_loader = DataLoader(train_set, batch_size=48, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=2, pin_memory=True, drop_last=True)

    model = SuperResolutionTransformer().to(device)
    content_discriminator = ContentDiscriminator()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(content_discriminator.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_generator_losses = []
    train_discriminator_losses = []

    valid_generator_losses = []
    valid_discriminator_losses = []

    model.to(device)
    content_discriminator.to(device)

    for epoch in range(epochs):
        model.train()
        content_discriminator.train()

        train_generator_loss = 0.0
        train_discriminator_loss = 0.0

        valid_generator_loss = 0.0
        valid_discriminator_loss = 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            optimizer2.zero_grad()

            upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

            output = model(data.to(device)).detach()

            target_discriminator_data = upsample(data.to(device).detach()) + target.to(device)
            target_content = content_discriminator(target_discriminator_data)

            discriminator_data = upsample(data.to(device).detach()) + output.to(device)
            up_scaled_content = content_discriminator(discriminator_data)

            discriminator_loss = content_discriminator_loss(criterion, up_scaled_content, target_content)

            discriminator_loss.backward(retain_graph=True)
            optimizer2.step()
            output = model(data.to(device))
            discriminator_data = upsample(data.to(device)) + output.to(device)

            up_scaled_content = content_discriminator(discriminator_data)

            generator_loss_result = generator_loss(criterion, up_scaled_content, output, target.to(device))

            generator_loss_result.backward()

            optimizer.step()

            train_generator_loss += generator_loss_result.item() * data.size(0)
            train_discriminator_loss += discriminator_loss.item() * data.size(0)

        model.eval()
        content_discriminator.eval()

        for data, target in valid_loader:
            output = model(data.to(device)).detach()
            loss = criterion(output, target.to(device))

            target_discriminator_data = upsample(data.to(device).detach()) + target.to(device)
            target_content = content_discriminator(target_discriminator_data)

            discriminator_data = upsample(data.to(device).detach()) + output.to(device)
            up_scaled_content = content_discriminator(discriminator_data)

            generator_loss_result = generator_loss(criterion, up_scaled_content, output, target.to(device))

            discriminator_loss = content_discriminator_loss(criterion, up_scaled_content, target_content)

            valid_generator_loss += generator_loss_result.item()*data.size(0)
            valid_discriminator_loss += discriminator_loss.item()*data.size(0)

        train_generator_loss = train_generator_loss / len(train_loader.sampler)
        train_discriminator_loss = train_discriminator_loss / len(train_loader.sampler)
        valid_generator_loss = valid_generator_loss / len(valid_loader.sampler)
        valid_discriminator_loss = valid_discriminator_loss / len(valid_loader.sampler)

        train_generator_losses.append(train_generator_loss)
        train_discriminator_losses.append(train_discriminator_loss)
        valid_generator_losses.append(valid_generator_loss)
        valid_discriminator_losses.append(valid_discriminator_loss)

        if epoch % print_result_per_epochs == 0:
            print('Epoch: {} \tTraining Generator Loss: {:.6f}  \tValidation Generator Loss: {:.6f} \tTraining '
                  'Discriminator Loss:{:.6f} \tValidation Discriminator Loss: {:.6f}'.format(
                epoch, train_generator_loss, valid_generator_loss, train_discriminator_loss,
                valid_discriminator_loss))

            model.eval()

            model.to(device)

            for batch_id, (x, y) in enumerate(train_loader):
                source = post_processed_image_from_torchtensor(x)
                target = post_processed_image_from_torchtensor(y)

                model.to(device)
                output = model(x.to(device))
                result = post_processed_image_from_torchtensor(output.to(cpu_device).detach())

                _, axs = plt.subplots(1, 3, figsize=(16, 4))
                axs = axs.flatten()
                axs[0].imshow(source / 255.0, )
                axs[1].imshow(target / 255.0, )
                axs[2].imshow(np.clip(result / 255.0, 0.0, 1.0), )

                if not os.path.isdir(out_path):
                    os.mkdir(out_path)

                plt.savefig(os.path.join(out_path, 'epoch-{}-out.png'.format(epoch)))

