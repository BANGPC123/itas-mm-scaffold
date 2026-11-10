import torch
from src.perception.ufld_v2 import local_expectation, plausible_lane


def test_vector_decode_matches_reference_including_grid_edges():
    torch.manual_seed(17)
    logits = torch.randn(1, 20, 12, 4)
    logits[0, 0, 0, 0] = 20
    logits[0, -1, 1, 0] = 20
    actual = local_expectation(logits)
    for anchor in range(12):
        for lane in range(4):
            idx = int(logits[0, :, anchor, lane].argmax())
            cells = torch.arange(max(0, idx-1), min(19, idx+1)+1)
            expected = ((logits[0, cells, anchor, lane].softmax(0)*cells).sum()+.5)/19
            torch.testing.assert_close(actual[anchor, lane], expected)


def test_rejects_horizontal_zigzag_across_vehicles():
    assert not plausible_lane([(800,290),(840,280),(890,295),(940,280),(980,340),(1050,355)],1280,468)


def test_accepts_sloped_road_boundary_but_not_short_fragment():
    assert plausible_lane([(500,220),(475,280),(440,340),(400,430)],1280,468)
    assert not plausible_lane([(500,420),(490,425),(485,430)],1280,468)
