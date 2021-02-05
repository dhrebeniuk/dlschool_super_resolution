import torch


def content_discriminator_loss(loss, disc_real_output, disc_generated_output):
    real_loss = loss(torch.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss(torch.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = (real_loss + generated_loss) / 2

    return total_disc_loss


def generator_loss(loss, disc_generated_output, gen_output, target):
    gan_loss = loss(torch.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = torch.mean(torch.abs(target - gen_output))
    lambda_value = 100
    total_gen_loss = gan_loss + (lambda_value * l1_loss)

    return total_gen_loss
