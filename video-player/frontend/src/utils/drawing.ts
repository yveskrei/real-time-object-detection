import type { BBox } from '../types';

export const COLORS: Record<string, string> = {
    person: '#0096FF', // Bright blue
    car: '#00C800',    // Green
    truck: '#FF6400',  // Orange
    dog: '#C800C8',    // Magenta
    cat: '#FF0064',    // Pink-red
    default: '#00C8C8' // Cyan
};

export const drawBBoxes = (
    ctx: CanvasRenderingContext2D,
    bboxes: BBox[],
    originalWidth: number,
    originalHeight: number,
    width: number,
    height: number,
    minConfidence: number
) => {
    if (originalWidth === 0 || originalHeight === 0 || width === 0 || height === 0) {
        return;
    }

    const scaleX = width / originalWidth;
    const scaleY = height / originalHeight;

    ctx.save();

    bboxes.forEach(bbox => {
        if (bbox.confidence < minConfidence) return;

        const color = COLORS[bbox.class_name.toLowerCase()] || COLORS.default;

        // Convert 1D indices to 2D coordinates
        const y1_orig = Math.floor(bbox.top_left_corner / originalWidth);
        const x1_orig = bbox.top_left_corner % originalWidth;
        const y2_orig = Math.floor(bbox.bottom_right_corner / originalWidth);
        const x2_orig = bbox.bottom_right_corner % originalWidth;

        // Scale to current display size
        const x1 = x1_orig * scaleX;
        const y1 = y1_orig * scaleY;
        const x2 = x2_orig * scaleX;
        const y2 = y2_orig * scaleY;

        const w = x2 - x1;
        const h = y2 - y1;

        // Draw Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, w, h);

        // Draw Label Background
        const label = `${bbox.class_name} ${bbox.confidence.toFixed(2)}`;
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 14;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight - 8, textWidth + 8, textHeight + 8);

        // Draw Text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x1 + 4, y1 - 6);
    });

    ctx.restore();
};
