from pathlib import Path
import pytest
from src.reasoning.offline_legal_retrieval import appendix_item, with_appendix_locator, OfflineLegalRetriever

ROOT = Path(__file__).resolve().parents[2]


def test_appendix_item_prefers_item_heading_over_figure_caption():
    text = 'Hình B.30 - Biển số P.130 ... B.30 Biển số P.130 "Cấm dừng xe và đỗ xe"'
    assert appendix_item(text, 'P.130') == ('B', 30, False)
    assert appendix_item('Hình B.30 - Biển số P.130', 'P.130') == ('B', 30, True)


def test_appendix_item_does_not_confuse_neighbouring_codes():
    text = 'B.31 Biển số P.131 (a,b,c) "Cấm đỗ xe"'
    assert appendix_item(text, 'P.130') is None
    assert appendix_item(text, 'P.131a') == ('B', 31, False)
    assert appendix_item(text, 'P.131d') is None
    assert appendix_item('B.23 Biển số P.123b', 'P.123b') == ('B', 23, False)
    sibling = 'B.24 Biển số P.124 (a,b) "Cấm quay đầu xe", biển số P.124c "...", biển số P.124d "..."'
    assert appendix_item(sibling, 'P.124d') == ('B', 24, False)
    assert appendix_item('E.7 Biển số I.407(a,b,c) "Đường một chiều"', 'I.407a') == ('E', 7, False)
    assert appendix_item('D.2 Biển sốR.302 (a,b,c)', 'R.302a') == ('D', 2, False)


def test_inherited_article_locator_is_replaced_only_with_evidence():
    hit = {'locator': 'Điều 84. Tổ chức thực hiện', 'excerpt': 'B.30 Biển số P.130 "Cấm dừng xe và đỗ xe"'}
    fixed = with_appendix_locator(hit, 'P.130')
    assert fixed['locator'] == 'Phụ lục B, mục B.30 (biển số P.130)'
    assert fixed['corpus_locator'] == 'Điều 84. Tổ chức thực hiện'
    unchanged = with_appendix_locator({'locator': 'Điều 22', 'excerpt': 'nhắc tới P.130'}, 'P.130')
    assert unchanged['locator'] == 'Điều 22' and unchanged['locator_source'] == 'corpus_heading'


@pytest.mark.skipif(not (ROOT / 'data/regulations/corpus.json').is_file(), reason='structured corpus absent')
def test_p130_lookup_no_longer_cites_final_article():
    hit = OfflineLegalRetriever().lookup_sign('P.130')
    assert hit and not hit['locator'].startswith('Điều 84')
    assert hit['locator'].startswith('Phụ lục B, mục B.30')
