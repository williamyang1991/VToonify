## saved models

folder structure (* model that is only for training, not required for inference):
```
checkpoint
|--encoder.pt                             % Pixel2style2pixel model
|--faceparsing.pt                         % Face parsing model
|--stylegan2-ffhq-config-f.pt             % * StyleGAN model
|--derections.npy                         % * Editing vectors
% the followings are VToonify models
|--vtoonify_t_cartoon
    |--pretrain.pt                        % * Pre-trained encoder for Cartoon style
    |--vtoonify.pt                        % VToonify-T model for Cartoon style
|--vtoonify_d_cartoon
    |--pretrain.pt                        % * Pre-trained encoder for Cartoon style
    |--exstyle_code.npy                   % Extrinsic style codes of Cartoon dataset
    |--vtoonify_s_d.pt                    % VToonify-Dsd model for Cartoon style
    |--vtoonify_s_d_c.pt                  % VToonify-Dsdc model for Cartoon style
    |--vtoonify_s026_d0.5.pt              % VToonify-D model for the 26th Cartoon style with style degree 0.5
    ...                                   % VToonify-D model for other settings
|--vtoonify_t_caricature
    % the same files as in vtoonify_t_cartoon
|--vtoonify_d_caricature
    % the same files as in vtoonify_d_cartoon
...
% the followings are pre-trained Toonify and DualStyleGAN models
|--cartoon
    |--generator.pt                       % * DualStyleGAN model
    |--exstyle_code.npy                   % * Extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy           % * Refined extrinsic style codes of Cartoon dataset
    |--finetune-000600.pt                 % * Toonify model (fine-tuned StyleGAN on Cartoon dataset)
|--caricature
    % the same files as in cartoon
...
```
