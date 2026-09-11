import os


def test_best_quality_all_nf_contains_default_and_all_neuralforecast_models(monkeypatch, tmp_path):
    from autogluon.timeseries.configs import get_hyperparameter_presets, get_predictor_presets
    from autogluon.timeseries.models.neuralforecast import NEURALFORECAST_MODELS, NFNHITSModel

    hyperparameter_presets = get_hyperparameter_presets()
    predictor_presets = get_predictor_presets()

    all_nf = hyperparameter_presets["default_all_nf"]
    default = hyperparameter_presets["default"]

    assert set(default).issubset(all_nf)
    assert {f"NF{name}" for name in NEURALFORECAST_MODELS}.issubset(all_nf)
    assert len({name for name in all_nf if name.startswith("NF")}) == len(NEURALFORECAST_MODELS) == 66
    assert all_nf["Chronos2"] == default["Chronos2"]

    preset = predictor_presets["best_quality_all_nf"]
    assert preset == {
        "hyperparameters": "default_all_nf",
        "num_val_windows": "auto",
        "refit_every_n_windows": "auto",
    }

    backend = tmp_path / "nf-python"
    backend.touch()
    monkeypatch.setenv("AUTOGLUON_NF_PYTHON", os.fspath(backend))
    model = NFNHITSModel(path=str(tmp_path / "model"), freq="D", prediction_length=2)
    assert model.get_hyperparameters()["python_executable"] == os.fspath(backend)
