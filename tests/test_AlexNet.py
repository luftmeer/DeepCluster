from deepcluster.AlexNet import AlexNet


def test_AlexNet_init():
    model = AlexNet()
    assert type(model) == AlexNet