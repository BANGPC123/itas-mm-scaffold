from src.context.sign_applicability import SignApplicability
from src.interaction.sign_speech import sign_speech
from src.utils.types import LaneInfo,SignDetection


def test_truck_supplement_and_adjacent_overhead_are_not_spoken():
    signs=[SignDetection('P.130',.9,(100,100,200,200),panel_role='primary'),
           SignDetection('truck',.9,(110,205,190,240),panel_role='supplementary',allowed_vehicles=['truck']),
           SignDetection('P.102',.9,(500,20,550,70),overhead=True)]
    lane=LaneInfo([],ego_boundaries=[[(200,300),(200,599)],[(400,300),(400,599)]])
    result=SignApplicability().apply(signs,lane,(600,800,3),'car')
    assert sign_speech(result)==''


def test_unknown_supplement_cannot_be_spoken_as_applicable():
    signs=[SignDetection('P.130',.9,(100,100,200,200),panel_role='primary',overhead=True),
           SignDetection('unknown',.9,(110,205,190,240),panel_role='supplementary')]
    lane=LaneInfo([],ego_boundaries=[[(50,300),(50,599)],[(250,300),(250,599)]])
    result=SignApplicability().apply(signs,lane,(600,800,3))
    assert result[0].applicability=='uncertain'
    assert sign_speech(result)==''  # unresolved supplementary plate must not produce voice


def test_known_name_is_spoken_without_inventing_speed_value():
    text=sign_speech([SignDetection('P.127',.9,(0,0,20,20))])
    assert text == ''  # a plausible but unverified code cannot trigger TTS


def test_detector_order_cannot_bypass_speech_cooldown():
    signs=[SignDetection('P.130',.9,(0,0,20,20)),SignDetection('P.131a',.8,(30,0,50,20))]
    assert sign_speech(signs)==sign_speech(list(reversed(signs)))
