/**
 * ScannedSurfaceFilter
 * 
 * Provides an object-attached holographic overlay visualization.
 * Driven by accepted frames and estimated observed surface area.
 * 
 * NOTE: This is an estimated visualization, not exact reconstructed geometry.
 */
class ScannedSurfaceFilter {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.width = 0;
        this.height = 0;
        this.lastUpdate = 0;
        this.fpsLimit = 15;
        this.rippleTime = 0;
        this.rippleOpacity = 0;
        this.scanOffset = 0;
        
        this.externalMask = null;
        this.maskConfidence = 0;
        this.maskMinConfidence = 0.75;
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.width = rect.width;
        this.height = rect.height;
    }

    /**
     * Update the overlay based on current state
     */
    update(azimuth, tilt, profile, summary, isAcceptedFrame) {
        if (!this.ctx) return;

        const now = Date.now();
        if (now - this.lastUpdate < 1000 / this.fpsLimit) return;
        this.lastUpdate = now;

        this.ctx.clearRect(0, 0, this.width, this.height);

        // If confidence is low (no summary or not recording), keep it subtle or hidden
        if (!summary || summary.percent === undefined) {
            this.canvas.style.opacity = "0";
            return;
        }
        this.canvas.style.opacity = "0.8";

        const silhouette = this.getSilhouettePath(profile);
        if (!silhouette) return;

        this.drawBaseEffect(silhouette);
        this.drawScannedArea(silhouette, azimuth, tilt, profile, summary);
        this.drawScanLines(silhouette);
        
        if (isAcceptedFrame) {
            this.rippleTime = 1.0;
            this.rippleOpacity = 0.5;
        }

        if (this.rippleTime > 0) {
            this.drawRipple(silhouette);
            this.rippleTime -= 0.05;
            this.rippleOpacity -= 0.025;
        }

        this.scanOffset = (this.scanOffset + 2) % 100;
    }

    /**
     * Set an external mask received from the backend (e.g. SAM2/SAM3)
     */
    setExternalMask(maskData) {
        if (!maskData) {
            this.externalMask = null;
            this.maskConfidence = 0;
            return;
        }

        this.externalMask = maskData;
        this.maskConfidence = maskData.confidence || 0;
    }

    getSilhouettePath(profile) {
        // Prefer high-confidence external mask
        if (this.externalMask && this.externalMask.mask && this.maskConfidence >= this.maskMinConfidence) {
            const path = new Path2D();
            const mask = this.externalMask.mask;
            const mw = this.externalMask.mask_width;
            const mh = this.externalMask.mask_height;
            
            // Scale mask to current canvas size
            const scaleX = this.width / mw;
            const scaleY = this.height / mh;

            if (this.externalMask.mask_format === 'polygon' && Array.isArray(mask)) {
                if (mask.length > 0) {
                    path.moveTo(mask[0][0] * scaleX, mask[0][1] * scaleY);
                    for (let i = 1; i < mask.length; i++) {
                        path.lineTo(mask[i][0] * scaleX, mask[i][1] * scaleY);
                    }
                    path.closePath();
                    return path;
                }
            }
            // Add other formats (RLE etc) if implemented later
        }

        const cx = this.width / 2;
        const cy = this.height / 2;
        const size = Math.min(this.width, this.height) * 0.45;
        const path = new Path2D();

        if (profile === 'bottle') {
            const w = size * 0.5;
            const h = size * 1.2;
            path.roundRect(cx - w/2, cy - h/2 + w*0.4, w, h - w*0.8, [20]); // Body
            path.arc(cx, cy - h/2 + w*0.4, w/2.5, Math.PI, 0); // Cap top
            path.roundRect(cx - w/2.2, cy + h/2 - w*0.4, w*0.9, w*0.3, [10]); // Base
        } else if (profile === 'box') {
            const w = size * 0.9;
            const h = size * 0.9;
            path.roundRect(cx - w/2, cy - h/2, w, h, [12]);
        } else {
            // Generic silhouette (soft capsule)
            const w = size * 0.7;
            const h = size * 1.0;
            path.roundRect(cx - w/2, cy - h/2, w, h, [40]);
        }
        return path;
    }

    drawBaseEffect(path) {
        this.ctx.save();
        // Subtle inner glow
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = 'rgba(0, 255, 204, 0.1)';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
        this.ctx.fill(path);
        
        // Thin border
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke(path);
        this.ctx.restore();
    }

    drawScannedArea(path, azimuth, tilt, profile, summary) {
        this.ctx.save();
        this.ctx.clip(path);

        if (profile === 'bottle') {
            this.renderBottleMapping(azimuth, summary.sectors);
        } else if (profile === 'box') {
            this.renderBoxMapping(azimuth, tilt, summary.sectors);
        } else {
            this.renderGenericMapping(summary.percent);
        }

        this.ctx.restore();
    }

    renderBottleMapping(azimuth, sectors) {
        if (!sectors) return;
        const numSectors = sectors.length;
        const cx = this.width / 2;
        const silhouetteWidth = Math.min(this.width, this.height) * 0.45 * 0.5;

        sectors.forEach((covered, i) => {
            if (!covered) return;
            
            let sectorAngle = (i * (360 / numSectors));
            let diff = sectorAngle - azimuth;
            while (diff > 180) diff -= 360;
            while (diff < -180) diff += 360;

            // Visibility filter (approx 160 degrees)
            if (Math.abs(diff) < 80) {
                const xPos = cx + (diff / 80) * (silhouetteWidth / 2);
                const w = (silhouetteWidth / (numSectors / 2)) * 1.8;
                
                const grad = this.ctx.createLinearGradient(xPos - w/2, 0, xPos + w/2, 0);
                grad.addColorStop(0, 'rgba(0, 255, 204, 0)');
                grad.addColorStop(0.5, 'rgba(0, 255, 204, 0.25)');
                grad.addColorStop(1, 'rgba(0, 255, 204, 0)');
                
                this.ctx.fillStyle = grad;
                this.ctx.fillRect(xPos - w/2, 0, w, this.height);
            }
        });
    }

    renderBoxMapping(azimuth, tilt, sectors) {
        // For box, we map the 4 main azimuth sectors to screen quadrants/bands
        // plus top/bottom based on tilt
        const cx = this.width / 2;
        const cy = this.height / 2;
        const w = Math.min(this.width, this.height) * 0.45 * 0.9;
        
        const faces = ['front', 'right', 'back', 'left'];
        faces.forEach((face, i) => {
            const sectorIdx = Math.floor(i * (sectors.length / 4));
            if (sectors[sectorIdx]) {
                let faceAngle = i * 90;
                let diff = faceAngle - azimuth;
                while (diff > 180) diff -= 360;
                while (diff < -180) diff += 360;

                if (Math.abs(diff) < 90) {
                    const xPos = cx + (diff / 90) * (w / 2);
                    const faceWidth = w / 2;
                    this.ctx.fillStyle = 'rgba(0, 255, 204, 0.15)';
                    this.ctx.fillRect(xPos - faceWidth/2, cy - w/2, faceWidth, w);
                }
            }
        });
    }

    renderGenericMapping(percent) {
        // Generic uses a simple horizontal scan fill based on percentage
        const h = Math.min(this.width, this.height) * 0.45;
        const fillH = h * (percent / 100);
        const cy = this.height / 2;
        
        const grad = this.ctx.createLinearGradient(0, cy + h/2, 0, cy + h/2 - fillH);
        grad.addColorStop(0, 'rgba(0, 255, 204, 0.2)');
        grad.addColorStop(1, 'rgba(0, 255, 204, 0.05)');
        
        this.ctx.fillStyle = grad;
        this.ctx.fillRect(0, cy + h/2 - fillH, this.width, fillH);
    }

    drawScanLines(path) {
        this.ctx.save();
        this.ctx.clip(path);
        this.ctx.strokeStyle = 'rgba(0, 255, 204, 0.05)';
        this.ctx.lineWidth = 1;
        
        const step = 8;
        const offset = (Date.now() / 50) % step;
        
        this.ctx.beginPath();
        for (let y = 0; y < this.height; y += step) {
            this.ctx.moveTo(0, y + offset);
            this.ctx.lineTo(this.width, y + offset);
        }
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawRipple(path) {
        this.ctx.save();
        this.ctx.clip(path);
        const progress = 1.0 - this.rippleTime;
        const radius = Math.max(this.width, this.height) * progress;
        
        this.ctx.beginPath();
        this.ctx.arc(this.width/2, this.height/2, radius, 0, Math.PI * 2);
        this.ctx.strokeStyle = `rgba(0, 255, 204, ${this.rippleOpacity})`;
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        this.ctx.restore();
    }
}

window.ScannedSurfaceFilter = ScannedSurfaceFilter;
