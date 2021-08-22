from model.encoder import VidEncoderModel
from model.decoder import Vid2FMRIModel


def get_model(backbone_name, embed_size, output_size, linear_pool=None, adaptive_pool=1, rnn_features=False, **kwargs):
    encoder = VidEncoderModel(embed_size=embed_size, backbone_name=backbone_name, linear_pool=linear_pool, adaptive_pool=adaptive_pool)
    model = Vid2FMRIModel(encoder, output_size=output_size, rnn_features=rnn_features)
    return model