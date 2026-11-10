import { useEffect, useRef } from "react";
import { cropRect } from "./logic.js";

// Enlarged view of one detected sign, cut from the annotated result image.
export default function SignThumb({ src, box, frameShape, size = 56 }) {
  const canvas = useRef(null);
  useEffect(() => {
    if (!src || !box) return undefined;
    let alive = true;
    const image = new Image();
    image.onload = () => {
      const element = canvas.current;
      if (!alive || !element) return;
      const rect = cropRect(box, image.naturalWidth, image.naturalHeight, frameShape?.[1], frameShape?.[0]);
      if (!rect) return;
      const scale = window.devicePixelRatio || 1;
      element.width = size * scale;
      element.height = size * scale;
      const context = element.getContext("2d");
      context.imageSmoothingQuality = "high";
      context.drawImage(image, rect[0], rect[1], rect[2], rect[3], 0, 0, element.width, element.height);
    };
    image.src = src;
    return () => {
      alive = false;
    };
  }, [src, box, frameShape, size]);
  if (!box) return <span className="thumb thumb-empty" style={{ width: size, height: size }} aria-hidden="true" />;
  return <canvas ref={canvas} className="thumb" style={{ width: size, height: size }} aria-hidden="true" />;
}
