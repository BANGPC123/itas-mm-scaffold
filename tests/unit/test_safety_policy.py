"""Safety invariants for the two-model / three-rule / Vietnamese-TTS pipeline.

These are software policy checks, NOT measured field accuracy or legal approval.
"""
from src.context.sign_applicability import SignApplicability
from src.interaction.sign_speech import sign_speech
from src.utils.types import LaneInfo, SignDetection


def reviewed(code='P.130', **kwargs):
    return SignDetection(code, .99, (270, 50, 320, 110),
                         panel_role='primary', label_verified=True, context_verified=True,
                         placement='roadside', lane_scope='all', **kwargs)


def lanes():
    return LaneInfo([], ego_boundaries=[
        [(200, 350), (200, 480), (200, 599)],
        [(400, 350), (400, 480), (400, 599)]
    ], backend='ufld_v2')


def test_unverified_automatic_detection_not_audible_despite_fake_overhead():
    guess=SignDetection('P.130', .99, (270, 50, 320, 110), overhead=True)
    result=SignApplicability().apply([guess],lanes(),(600, 800, 3))
    assert result[0].applicability=='uncertain'
    assert sign_speech(result)==''


def test_verified_roadside_car_generates_short_speech():
    result=SignApplicability().apply([reviewed()],LaneInfo([]),(600, 800, 3))
    assert result[0].applicability=='applies'
    assert sign_speech(result)=='Biển áp dụng cho làn đang đi: Cấm dừng xe và đỗ xe.'


def test_just_vehicle_scope_without_label_verification_is_not_ignored():
    fake=SignDetection('P.130', .9, (100, 100, 200, 200),
                       panel_role='primary',allowed_vehicles=['motorcycle'])
    result=SignApplicability().apply([fake],LaneInfo([]),(600,800,3),vehicle='car')
    assert result[0].applicability=='uncertain'


def test_verified_overhead_geometry_and_lane_scope_must_agree():
    s=SignDetection(
        'P.130', .99, (270, 50, 320, 110), panel_role='primary',
        placement='overhead',lane_scope='ego',label_verified=True,context_verified=True)
    accepted=SignApplicability().apply([s],lanes(),(600,800,3))
    assert accepted[0].applicability=='applies'
    other=SignDetection('P.130', .99, (480, 50, 520, 110),panel_role='primary',
                        placement='overhead',lane_scope='ego',label_verified=True,context_verified=True)
    denied=SignApplicability().apply([other],lanes(),(600,800,3))
    assert denied[0].applicability=='uncertain'
    assert sign_speech(denied)==''


def test_verified_compound_vehicle_scope_preserved():
    primary=SignDetection('P.130', .9, (100, 100, 200, 200),
                          panel_role='primary',placement='roadside',lane_scope='all',
                          label_verified=True,context_verified=True)
    plate=SignDetection('truck panel', .9, (115, 205, 185, 240),
                        panel_role='supplementary',allowed_vehicles=['truck'],
                        label_verified=True,context_verified=True)
    result=SignApplicability().apply([primary,plate],LaneInfo([]),(600,800,3),'car')
    assert len(result)==1 and result[0].is_compound
    assert result[0].applicability=='ignored'
    assert sign_speech(result)==''
    result=SignApplicability().apply([primary,plate],LaneInfo([]),(600,800,3),'truck')
    assert result[0].applicability=='applies'
    assert 'xe tải' in sign_speech(result)


def test_unverified_plate_can_never_suppress_or_trigger_command():
    primary=SignDetection('P.130', .9, (100, 100, 200, 200),
                          panel_role='primary',placement='roadside',lane_scope='all',
                          label_verified=True,context_verified=True)
    plate=SignDetection('truck panel', .9, (115,205,185,240),
                        panel_role='supplementary',allowed_vehicles=['truck'],
                        context_verified=True,label_verified=False)
    result=SignApplicability().apply([primary,plate],LaneInfo([]),(600,800,3),'car')
    assert result[0].applicability=='uncertain'
    assert sign_speech(result)==''


def test_numeric_sign_family_has_no_invented_value_in_audio():
    for code in ('P.127','P.117'):
        assert sign_speech(SignApplicability().apply(
            [reviewed(code)],LaneInfo([]),(600,800,3)))==''


def test_unrecognized_but_confident_class_never_announced():
    unknown=SignDetection('advertisement',1.,(100,100,200,200),panel_role='primary',
                          placement='roadside',lane_scope='all',
                          context_verified=True,label_verified=True)
    result=SignApplicability().apply([unknown],LaneInfo([]),(600,800,3))
    assert result[0].applicability=='uncertain'
    assert sign_speech(result)==''


def test_unknown_small_panel_below_primary_becomes_unresolved_compound():
    main=SignDetection('P.130',.96,(100,100,200,200),panel_role='unknown')
    plate=SignDetection('unknown',.77,(120,210,180,244),panel_role='unknown')
    grouped=SignApplicability().apply([main,plate],LaneInfo([]),(600,800,3))
    assert len(grouped)==1
    assert grouped[0].is_compound
    assert grouped[0].panel_labels==['P.130','unknown']
    assert grouped[0].panel_role_source=='geometric_candidate'
    assert grouped[0].applicability=='uncertain'
    assert not main.is_compound and main.panel_role=='unknown'
    assert sign_speech(grouped)==''


def test_ambiguous_plate_does_not_auto_attach():
    a=SignDetection('P.130',.9,(100,100,180,200))
    b=SignDetection('P.131a',.9,(150,100,230,200))
    sub=SignDetection('unknown',.9,(154,208,178,237))
    results=SignApplicability().group([a,b,sub])
    assert len(results)==3 and not any(s.is_compound for s in results)


def test_lateral_or_large_unknown_object_is_not_inferred_as_plate():
    a=SignDetection('P.130',.9,(100,100,200,200))
    side=SignDetection('unknown',.9,(210,210,260,240))
    large=SignDetection('unknown',.9,(120,210,180,335))
    results=SignApplicability().group([a,side,large])
    assert len(results)==3
    assert not any(s.is_compound for s in results)
