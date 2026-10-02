/**
 * Keeps faces identifiable from one camera frame to the next.
 *
 * Each frame's detections are matched to the faces seen before by box overlap,
 * so a face keeps its number while it moves. Probabilities and boxes are
 * smoothed with an exponential average: the classifier is sensitive to a few
 * pixels of crop, and without smoothing the reading would flicker on every frame.
 * A face that goes undetected for a few frames is kept (marked stale) instead
 * of vanishing and coming back with a new number.
 */

/** @typedef {{ x: number, y: number, w: number, h: number }} Rect */

export function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = w * h;
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

const mix = (old, fresh, alpha) => alpha * fresh + (1 - alpha) * old;

/**
 * @param {{ iouMin?: number, alpha?: number, maxMissed?: number }} [options]
 *   iouMin: minimum overlap to call two boxes the same face.
 *   alpha: weight of the newest frame in the averages (1 means no smoothing).
 *   maxMissed: frames a face may go undetected before it is dropped.
 */
export function createTracker({ iouMin = 0.25, alpha = 0.5, maxMissed = 3 } = {}) {
  let nextId = 1;
  let tracks = [];

  return {
    /**
     * @param {Array<{ box: Rect, probabilities: Record<string, number>, crop: object, heatmap: any, cam: any }>} faces
     * @returns the live tracks, ordered by id, each with the smoothed box, probabilities, emotion and confidence
     */
    update(faces) {
      const pairs = [];
      tracks.forEach((track, t) => faces.forEach((face, f) => {
        const overlap = iou(track.box, face.box);
        if (overlap >= iouMin) pairs.push({ t, f, overlap });
      }));
      pairs.sort((a, b) => b.overlap - a.overlap);
      const takenTrack = new Set();
      const takenFace = new Set();
      for (const { t, f } of pairs) {
        if (takenTrack.has(t) || takenFace.has(f)) continue;
        takenTrack.add(t);
        takenFace.add(f);
        const track = tracks[t];
        const face = faces[f];
        const probabilities = {};
        for (const key of Object.keys(face.probabilities)) probabilities[key] = mix(track.probabilities[key] ?? 0, face.probabilities[key], alpha);
        track.box = {
          x: mix(track.box.x, face.box.x, alpha),
          y: mix(track.box.y, face.box.y, alpha),
          w: mix(track.box.w, face.box.w, alpha),
          h: mix(track.box.h, face.box.h, alpha),
        };
        Object.assign(track, summarize(probabilities), { crop: face.crop, heatmap: face.heatmap, cam: face.cam, missed: 0, stale: false });
      }

      tracks.forEach((track, t) => {
        if (takenTrack.has(t)) return;
        track.missed += 1;
        track.stale = true;
      });
      tracks = tracks.filter((track) => track.missed <= maxMissed);

      faces.forEach((face, f) => {
        if (takenFace.has(f)) return;
        tracks.push({
          id: nextId++,
          box: { ...face.box },
          crop: face.crop,
          heatmap: face.heatmap,
          cam: face.cam,
          ...summarize({ ...face.probabilities }),
          missed: 0,
          stale: false,
        });
      });
      tracks.sort((a, b) => a.id - b.id);
      return tracks.map((track) => ({ ...track, box: { ...track.box } }));
    },
  };
}

function summarize(probabilities) {
  let emotion = null;
  for (const [key, p] of Object.entries(probabilities)) if (emotion === null || p > probabilities[emotion]) emotion = key;
  return { probabilities, emotion, confidence: probabilities[emotion] };
}
