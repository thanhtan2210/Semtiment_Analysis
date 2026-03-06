import pytest


def test_import_and_predict():
    import train  # local module in Semtiment_Analysis

    pipe = train.build_pipeline()
    # Pipeline should be able to fit tiny sample
    X = ["tuyệt vời", "tệ hại", "bình thường"]
    y = ["positive", "negative", "neutral"]
    pipe.fit(X, y)
    pred = pipe.predict(["sản phẩm này quá tốt"])
    assert pred[0] in {"positive", "negative", "neutral"}
