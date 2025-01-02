#############################################################################################################
#                                                                                                           #
#   Author      : Stefano Giacomelli                                                                        #
#   Affiliation : PhD candidate at University of L'Aquila (Italy)                                           #
#   Department  : DISIM - Department of Information Engineering, Computer Science and Mathematics           #
#   Description : Test function script (PyTest) for General Purpose (Audio Tagging) audio embedding models. #
#   Last Update : 2024-12-07                                                                                #
#                                                                                                           #
#############################################################################################################
import pytest
import torch
import numpy as np
from audio_embeddings import package_install_and_import, model_init, pre_process_input, compute_embedding

DEBUG = True
MODEL_NAMES = ['audio-mlp',
               'ced_base',
               'ced_mini',
               'ced_small',
               'ced_tiny',
               'epanns',
               'ecapa_tdnn',
               'wav2vec2_base',
               'wav2vec2_large', 
               'wav2vec2_large_lv60k', 
               'wav2vec2_xlsr53', 
               #'wav2vec2_xlsr_300m', 
               #'wav2vec2_xlsr_1b', 
               #'wav2vec2_xlsr_2b', 
               'hubert_base', 
               #'hubert_large', 
               #'hubert_xlarge', 
               'wavlm_base', 
               'wavlm_base_plus', 
               'wavlm_large',
               'panns_Cnn14',
               'panns_ResNet38',
               'panns_Wavegram_Logmel_Cnn14',
               'torchopenl3_env_linear_512',
               'torchopenl3_env_linear_6144',
               'torchopenl3_env_mel128_512',
               'torchopenl3_env_mel128_6144',
               'torchopenl3_env_mel256_512',
               'torchopenl3_env_mel256_6144',
               'passt_s_p16_s16_128_ap468',
               'passt_s_swa_p16_128_ap476',
               'passt_s_kd_p16_128_ap486',
               'passt_l_kd_p16_128_ap47',
               'vggish', 
               'wav2clip',
               #'yamnet'                     # Doesn't work with batches (juxtapose results) --> PreProcessor activity
               ]

test_file_path = './test_audio_44.1KHz_16bit.wav'


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_general_purpose_models(model_name):
    """
    Test general purposemodels runtime.

    :param model_name: Name of the model to test
    """
    try:
        # Install and import required packages
        imported_modules = package_install_and_import(f'general_purpose/{model_name}', debug=DEBUG)
        assert imported_modules is not None, f"Failed to import modules for {model_name}"

        # Initialize the model
        model = model_init(f'general_purpose/{model_name}', imported_modules, debug=DEBUG)
        assert model is not None, f"Model instance is None for {model_name}"

        # Setup model-specific inputs
        model_info = imported_modules.get("model_info", {})
        if model_name == 'vggish':
            input_set = [[np.random.uniform(-1, 1) for _ in range(160000)],
                         np.random.uniform(-1, 1, (160000)).astype(np.float32)]
        elif model_name.startswith('encodec_24'):
            input_set = [[[[np.random.uniform(-1, 1) for _ in range(240000)] for _ in range(1)] for _ in range(2)],
                         np.random.uniform(-1, 1, (2, 1, 240000)).astype(np.float32),
                         torch.rand((2, 1, 240000), dtype=torch.float32) * 2.0 - 1.0]
        elif model_name.startswith('encodec_48'):
            input_set = [[[[np.random.uniform(-1, 1) for _ in range(480000)] for _ in range(2)] for _ in range(2)],
                         np.random.uniform(-1, 1, (2, 2, 480000)).astype(np.float32),
                         torch.rand((2, 2, 480000), dtype=torch.float32) * 2.0 - 1.0]
        else:
            if model_info.get("sample_rate") == 16000:
                input_set = ['./test_audio_44.1KHz_16bit.wav',
                             [[np.random.uniform(-1, 1) for _ in range(160000)] for _ in range(2)],
                             np.random.uniform(-1, 1, (2, 160000)).astype(np.float32),
                             torch.rand((2, 160000), dtype=torch.float32) * 2.0 - 1.0]
            else:
                input_set = ['./test_audio_44.1KHz_16bit.wav',
                             [[np.random.uniform(-1, 1) for _ in range(320000)] for _ in range(2)],
                             np.random.uniform(-1, 1, (2, 320000)).astype(np.float32),
                             torch.rand((2, 320000), dtype=torch.float32) * 2.0 - 1.0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for input_data in input_set:
            # Pre-process input
            preprocessed_input = pre_process_input(f'general_purpose/{model_name}', input_data, imported_modules, device=device, debug=DEBUG)
            assert isinstance(preprocessed_input, torch.Tensor), f"Preprocessed input is not a torch.Tensor: {type(preprocessed_input)}"
            print(f"Model: {model_name}, Input Type: {type(input_data)}, Pre-processed Shape: {preprocessed_input.shape}")
            
            # Compute the embedding
            embedding = compute_embedding(f'general_purpose/{model_name}', imported_modules, model, preprocessed_input, debug=DEBUG)
            assert embedding is not None, f"Embedding is None for {model_name} with {preprocessed_input}"
            print(f"Model: {model_name}, Model Input Shape: {preprocessed_input.shape}, Embedding Shape: {embedding.shape}")
            
            del preprocessed_input, embedding
        del imported_modules, model
    
    except Exception as e:
        pytest.fail(f"Test failed for {model_name}: {e}")
