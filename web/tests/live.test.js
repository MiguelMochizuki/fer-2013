import test from "node:test";
import assert from "node:assert/strict";
import { createTracker } from "../src/live.js";

const EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
/** A detected face as the pipeline returns it, with all the probability on `emotion`. */
function face(x, y, emotion = "happy", w = 100, h = 100, spread = 0) {
  const probabilities = Object.fromEntries(EMOTIONS.map((e) => [e, e === emotion ? 1 - spread : spread / 6]));
  return { box: { x, y, w, h }, emotion, confidence: 1 - spread, probabilities, crop: { gray: new Uint8Array(4), w: 2, h: 2 }, heatmap: null, cam: null };
}

test("new faces get ids 1..n in input order", () => {
  const t = createTracker();
  const tracks = t.update([face(0, 0), face(300, 0)]);
  assert.deepEqual(tracks.map((x) => x.id), [1, 2]);
});

test("a face that moves a little keeps its id, whatever the input order", () => {
  const t = createTracker();
  t.update([face(0, 0), face(300, 0)]);
  const tracks = t.update([face(310, 4), face(6, 2)]); // swapped order, both moved
  assert.deepEqual(tracks.map((x) => x.id), [1, 2]);
  assert.ok(tracks[0].box.x < 100 && tracks[1].box.x > 200);
});

test("a new face gets a new id and ids are never reused", () => {
  const t = createTracker({ maxMissed: 1 });
  t.update([face(0, 0)]);
  t.update([]);
  t.update([]); // expired
  const tracks = t.update([face(0, 0)]);
  assert.deepEqual(tracks.map((x) => x.id), [2]);
});

test("probabilities are smoothed with an exponential average", () => {
  const t = createTracker({ alpha: 0.5 });
  t.update([face(0, 0, "happy")]);
  const [track] = t.update([face(0, 0, "sad")]);
  assert.ok(Math.abs(track.probabilities.happy - 0.5) < 1e-9);
  assert.ok(Math.abs(track.probabilities.sad - 0.5) < 1e-9);
});

test("one odd frame does not flip the emotion", () => {
  const t = createTracker({ alpha: 0.4 });
  for (let i = 0; i < 4; i++) t.update([face(0, 0, "happy")]);
  const [track] = t.update([face(0, 0, "sad")]);
  assert.equal(track.emotion, "happy");
  assert.ok(track.confidence > 0.5 && track.confidence <= 1);
});

test("a sustained change does flip the emotion", () => {
  const t = createTracker({ alpha: 0.5 });
  t.update([face(0, 0, "happy")]);
  let track;
  for (let i = 0; i < 4; i++) [track] = t.update([face(0, 0, "surprise")]);
  assert.equal(track.emotion, "surprise");
});

test("boxes are smoothed too", () => {
  const t = createTracker({ alpha: 0.5 });
  t.update([face(0, 0)]);
  const [track] = t.update([face(20, 10)]);
  assert.deepEqual([track.box.x, track.box.y], [10, 5]);
});

test("a missing face is kept as stale for a few frames, then dropped", () => {
  const t = createTracker({ maxMissed: 2 });
  t.update([face(0, 0)]);
  assert.equal(t.update([])[0].stale, true);
  assert.equal(t.update([]).length, 1);
  assert.equal(t.update([]).length, 0);
});

test("a stale track that reappears is fresh again", () => {
  const t = createTracker({ maxMissed: 3 });
  t.update([face(0, 0)]);
  t.update([]);
  const [track] = t.update([face(4, 4)]);
  assert.equal(track.id, 1);
  assert.equal(track.stale, false);
});

test("two faces never claim the same track", () => {
  const t = createTracker();
  t.update([face(0, 0)]);
  const tracks = t.update([face(2, 2), face(5, 5)]); // both overlap the old box
  assert.equal(new Set(tracks.map((x) => x.id)).size, 2);
});
