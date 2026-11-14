import torch

from kick_analysis import config
from kick_analysis.models import KickLSTM, load_checkpoint, quantize, save_checkpoint


def test_forward_shape():
    model = KickLSTM(config.IMPACT_INPUT_SIZE, config.HIDDEN_SIZE, len(config.IMPACT_LABELS))
    x = torch.randn(4, config.IMPACT_SEQ_LEN, config.IMPACT_INPUT_SIZE)
    assert model(x).shape == (4, 3)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = KickLSTM(config.IMPACT_INPUT_SIZE, config.HIDDEN_SIZE, len(config.IMPACT_LABELS)).eval()
    path = tmp_path / "impact.pt"
    save_checkpoint(model, config.IMPACT_LABELS, path)

    loaded, labels = load_checkpoint(path)
    assert labels == list(config.IMPACT_LABELS)
    x = torch.randn(2, config.IMPACT_SEQ_LEN, config.IMPACT_INPUT_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


def test_quantized_model_runs_and_stays_close(tmp_path):
    torch.manual_seed(0)
    model = KickLSTM(config.FOOT_INPUT_SIZE, config.HIDDEN_SIZE, len(config.FOOT_LABELS)).eval()
    path = tmp_path / "foot.pt"
    save_checkpoint(model, config.FOOT_LABELS, path)

    quantized, _ = load_checkpoint(path, quantized=True)
    x = torch.randn(2, config.FOOT_SEQ_LEN, config.FOOT_INPUT_SIZE)
    with torch.no_grad():
        original, approx = model(x), quantized(x)
    assert approx.shape == (2, 5)
    # int8로 바꾸면 값이 조금 달라지지만 크게 벗어나면 안 된다
    assert (original - approx).abs().max() < 0.1


def test_quantize_replaces_lstm_and_linear():
    model = KickLSTM(config.IMPACT_INPUT_SIZE, config.HIDDEN_SIZE, 3)
    quantized = quantize(model)
    assert "quantized" in type(quantized.lstm).__module__
    assert "quantized" in type(quantized.fc).__module__
