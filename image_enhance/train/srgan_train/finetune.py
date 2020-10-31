from train.srgan_train.face_dataset import FACE

from model.srgan.srgan import generator, discriminator
from train.srgan_train.train import SrganTrainer

train_loader = FACE(scale=4,  # 2, 3, 4 or 8
                     subset='train')  # Training dataset are images 001 - 800

# Create a tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16,  # batch size as described in the EDSR and WDSR papers
                                random_transform=True,  # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)  # repeat iterating over training images indefinitely


# Create a new generator and init it with pre-trained weights.
gan_generator = generator()
gan_generator.load_weights('weights/srgan/gan_generator.h5')
gan_discriminator = discriminator()

gan_discriminator.load_weights('weights/srgan/gan_discriminator.h5')
# Create a training context for the GAN (generator + discriminator).
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=gan_discriminator)

# Train the GAN with 200,000 steps.
gan_trainer.train(train_ds, steps=80000)

# Save weights of generator and discriminator.
#gan_trainer.generator.save_weights('weights/srgan/gan_generator.h5')
#gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator.h5')
