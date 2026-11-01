import test from 'node:test';
import assert from 'node:assert/strict';
import {formatTime,signGroup,badgeStyle,estimateRemaining,formatRemaining,videoRows,imageRows,hitLocator,shortExcerpt,errorMessage,cropRect} from './logic.js';

test('formatTime shows minutes and seconds', () => {
  assert.equal(formatTime(0), '00:00');
  assert.equal(formatTime(3.9), '00:03');
  assert.equal(formatTime(75), '01:15');
  assert.equal(formatTime(-2), '00:00');
  assert.equal(formatTime(undefined), '00:00');
});

test('sign codes are grouped by prefix and get a fixed badge colour', () => {
  assert.equal(signGroup('P.112'), 'prohibition');
  assert.equal(signGroup('DP.134'), 'prohibition');
  assert.equal(signGroup('W.245a'), 'warning');
  assert.equal(signGroup('R.302a'), 'mandatory');
  assert.equal(signGroup('I.407'), 'information');
  assert.equal(signGroup('S.505a'), 'other');
  assert.equal(signGroup(''), 'other');
  assert.equal(badgeStyle('P.112').background, '#c0392b');
  assert.equal(badgeStyle('W.245a').color, '#1f2937');
});

test('remaining time is estimated from the share of frames done', () => {
  assert.equal(estimateRemaining(60, 0, 100), null);
  assert.equal(estimateRemaining(60, 25, 100), 180);
  assert.equal(estimateRemaining(60, 100, 100), 0);
  assert.equal(estimateRemaining(60, 10, 0), null);
  assert.equal(formatRemaining(null), 'Đang tính thời gian còn lại…');
  assert.equal(formatRemaining(30), 'Còn khoảng 30 giây');
  assert.equal(formatRemaining(300), 'Còn khoảng 5 phút');
});

test('video rows list every code of each cue at its time', () => {
  const cues = [{t_seconds: 3.167, codes: ['P.112', 'W.245a']}, {t_seconds: 13.167, codes: ['P.117']}];
  const names = {'P.112': 'Cấm người đi bộ', 'W.245a': 'Đi chậm'};
  const rows = videoRows(cues, names);
  assert.deepEqual(rows.map(r => [r.time, r.code, r.name]), [
    ['00:03', 'P.112', 'Cấm người đi bộ'],
    ['00:03', 'W.245a', 'Đi chậm'],
    ['00:13', 'P.117', ''],
  ]);
  assert.deepEqual(videoRows(undefined, names), []);
});

test('image rows mark unrecognised panels and skip lookup for them', () => {
  const signs = [
    {label: 'P.112', name_vi: 'Cấm người đi bộ', confidence: 0.912, is_compound: false},
    {label: 'unknown', confidence: 0.6, is_compound: false},
    {label: '', confidence: 0.7, is_compound: true, panel_labels: ['W.201a', 'S.505a']},
  ];
  const rows = imageRows(signs, {});
  assert.equal(rows[0].code, 'P.112');
  assert.equal(rows[0].confidence, 91);
  assert.equal(rows[0].canLookUp, true);
  assert.equal(rows[1].unknown, true);
  assert.equal(rows[1].name, 'Chưa nhận dạng');
  assert.equal(rows[1].canLookUp, false);
  assert.equal(rows[2].code, 'W.201a + S.505a');
  assert.equal(rows[2].canLookUp, false);
});

test('legal hits show a locator and a short excerpt', () => {
  assert.equal(hitLocator({locator: 'Điều 5', page: 3}), 'Điều 5');
  assert.equal(hitLocator({page: 3}), 'trang PDF 3');
  assert.equal(hitLocator({}), 'đoạn nguồn');
  assert.equal(shortExcerpt('a   b\n c'), 'a b c');
  const long = 'x'.repeat(400);
  assert.equal(shortExcerpt(long, 100).length, 100);
  assert.ok(shortExcerpt(long, 100).endsWith('…'));
});

test('HTTP failures map to clear Vietnamese messages', () => {
  assert.match(errorMessage(0, '', 'x'), /Mất kết nối/);
  assert.match(errorMessage(409, 'Đang có một video', 'x'), /xử lý một video khác/);
  assert.match(errorMessage(429, '', 'x'), /đang bận/);
  assert.equal(errorMessage(415, 'Hiện chỉ hỗ trợ MP4.', 'x'), 'Hiện chỉ hỗ trợ MP4.');
  assert.equal(errorMessage(500, undefined, 'Lỗi chung'), 'Lỗi chung');
});

test('cropRect pads, scales and stays inside the image', () => {
  assert.deepEqual(cropRect([100, 100, 120, 120], 200, 200, 400, 400, 0), [50, 50, 10, 10]);
  const [x, y, w, h] = cropRect([0, 0, 40, 20], 1000, 500, 1000, 500, 0.5);
  assert.equal(w, 80); assert.equal(h, 80); assert.equal(x, 0); assert.equal(y, 0);
  assert.equal(cropRect(null, 10, 10, 10, 10), null);
  assert.equal(cropRect([5, 5, 5, 9], 10, 10, 10, 10), null);
});
