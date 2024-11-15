'''

Seed fixo!
------------------------------------------------------------------------
- N: 20, 50
- Learning Rate: 1e-5,
- Embedding: 64, 128, 256
- Batch Size: 32, 256, 512
- Net Type: Transformer, FC
- Extra Diffusion Steps: 0, 8, 16
- Beta range: (1e-4, 0.01), (1e-4, 0.02), (1e-4, 0.05)
- Beta scheduler: Linear, Cosine, Quadrativ
- Alpha: (Fixed: 0.8, 0.5, 0.3), Exponential, Cosine
- Com modelo pré-treinado e sem modelo pré-treinado
------------------------------------------------------------------------
- Avaliar a distância média percorrida antes de uma infração em 5 rodadas
partindo de um mesmo ponto

'''

from learn_bc import learn_bc, set_seed, reset_weights
import numpy as np
import itertools



# model = MyModel()
# reset_weights(model)  # Reinicializa os pesos para os mesmos valores


if __name__ == '__main__':
    device = 'cuda'

    pretrained_bc_list = [True, False]
    lrate_type = ['cosine', 'fixed']
    alpha_list = ['cosine', 'fixed_08', 'fixed_03', 'exponential']
    beta_scheduler_list = ["cosine", "linear", "quadratic"]
    betas_list = [(1e-4, 0.01), (1e-4, 0.02), (1e-4, 0.05)]
    n_T_list = [20, 50]
    lrate_list = [1e-5, 5e-4, 1e-3]
    net_type_list = ["transformer", "fc"]
    batch_size_list = [32, 512]
    embedding_dim_list = [64, 128]

    # Gerando as combinações
    params_product = itertools.product(
        pretrained_bc_list,
        lrate_type,
        alpha_list,
        beta_scheduler_list,
        betas_list,
        n_T_list,
        lrate_list,
        net_type_list,
        batch_size_list,
        embedding_dim_list,
    )
    params_list = list(params_product)


    params_grid = np.meshgrid(n_T_list,
                              lrate_list,
                              lrate_type,
                              net_type_list,
                            #   extra_diffusion_steps_list,
                              betas_list,
                              beta_scheduler_list,
                              alpha_list,
                              pretrained_bc_list)
    params_list = np.array(params_grid).T.reshape(-1, len(params_grid))
    stop=1

    set_seed(42)