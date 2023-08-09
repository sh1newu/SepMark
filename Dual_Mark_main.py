import yaml
from easydict import EasyDict
import os
import time
from shutil import copyfile
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network.Dual_Mark import *
from utils import *


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def main():

    seed_torch(42) # it doesnot work if the mode of F.interpolate is "bilinear"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/train_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    project_name = args.project_name
    epoch_number = args.epoch_number
    batch_size = args.batch_size
    lr = args.lr
    beta1 = args.beta1
    image_size = args.image_size
    message_length = args.message_length
    message_range = args.message_range
    attention_encoder = args.attention_encoder
    attention_decoder = args.attention_decoder
    weight = args.weight
    dataset_path = args.dataset_path
    save_images_number = args.save_images_number
    noise_layers_R = args.noise_layers.pool_R
    noise_layers_F = args.noise_layers.pool_F

    project_name += "_" + str(image_size) + "_" + str(message_length) + "_" + str(message_range) + "_" + str(lr) + "_" + str(beta1) + "_" + attention_encoder + "_" + attention_decoder
    for i in weight:
        project_name += "_" +  str(i)
    result_folder = "results/" + time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime()) + "/"
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    copyfile("cfg/train_DualMark.yaml", result_folder + "train_DualMark.yaml")
    writer = SummaryWriter('runs/'+ project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    network = Network(message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)

    train_dataset = attrsImgDataset(os.path.join(dataset_path, "train_" + str(image_size)), image_size, "celebahq")
    #train_dataset = maskImgDataset(os.path.join(dataset_path, "train_" + str(image_size)), image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = attrsImgDataset(os.path.join(dataset_path, "val_" + str(image_size)), image_size, "celebahq")
    #val_dataset = maskImgDataset(os.path.join(dataset_path, "val_" + str(image_size)), image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart training : \n\n")

    for epoch in range(1, epoch_number + 1):

        running_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        '''
        train
        '''
        for step, (image, mask) in enumerate(train_dataloader, 1):
            print(device)
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            result = network.train(image, message, mask)

            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(train_dataloader)))

            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Train/" + key, float(result[key]), (epoch - 1) * len(train_dataloader) + step)
                running_result[key] += float(result[key])

        '''
        train results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in running_result:
            content += key + "=" + str(running_result[key] / step) + ","
            writer.add_scalar("Train_epoch/" + key, float(running_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/train_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        validation
        '''

        val_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        saved_iterations = np.random.choice(np.arange(1, len(val_dataloader)+1), size=save_images_number, replace=False)
        saved_all = None

        for step, (image, mask) in enumerate(val_dataloader, 1):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            result, (images, encoded_images, noised_images) = network.validation(image, message, mask)

            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(val_dataloader)))
            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Val/" + key, float(result[key]), (epoch - 1) * len(val_dataloader) + step)
                val_result[key] += float(result[key])

            if step in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(image, encoded_images, noised_images)
                else:
                    saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        save_images(saved_all, epoch, result_folder + "images/", resize_to=None)

        '''
        validation results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in val_result:
            content += key + "=" + str(val_result[key] / step) + ","
            writer.add_scalar("Val_epoch/" + key, float(val_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/val_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        save model
        '''
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
        path_discriminator = path_model + "D_" + str(epoch) + ".pth"
        network.save_model(path_encoder_decoder, path_discriminator)

        writer.close()


if __name__ == '__main__':
    main()
