import React, { useEffect, useRef } from 'react';
import type { BBox } from '../types';
import { drawBBoxes } from '../utils/drawing';

interface BBoxOverlayProps {
    bboxes: BBox[];
    originalWidth: number;
    originalHeight: number;
    minConfidence: number;
    show: boolean;
    width: number;
    height: number;
    offsetX?: number;
    offsetY?: number;
}



export const BBoxOverlay: React.FC<BBoxOverlayProps> = ({
    bboxes,
    originalWidth,
    originalHeight,
    minConfidence,
    show,
    width,
    height,
    offsetX = 0,
    offsetY = 0
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d', { alpha: true });
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (!show) return;

        drawBBoxes(ctx, bboxes, originalWidth, originalHeight, width, height, minConfidence);

    }, [bboxes, originalWidth, originalHeight, minConfidence, show, width, height]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="absolute pointer-events-none"
            style={{
                left: `${offsetX}px`,
                top: `${offsetY}px`
            }}
        />
    );
};
