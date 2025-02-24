import argparse
import os
from os import makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_saving_dir', type=str, default=".")
    parser.add_argument('--n_dims', type=int, default=7)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset_size', type=int, default=64000)
    parser.add_argument('--parametrization_type', type=str, default="qr")
    parser.add_argument('--gpuid', type=int, default=0)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    
    from symmetry_gan import *
    g_model = define_generator(parametrization_type=args.parametrization_type)
    d_model = define_discriminator(n_dims=args.n_dims, learning_rate=args.lr_d) 
    gan_model = define_gan(g_model, d_model, learning_rate=args.lr_g)
    
    makedirs(args.model_saving_dir, exist_ok=True)
    
    train(g_model, 
          d_model,
          gan_model,
          n_epochs=args.n_epochs,
          batch_size=args.batch_size,
          dataset_size=args.dataset_size,
          saving_dir=args.model_saving_dir)