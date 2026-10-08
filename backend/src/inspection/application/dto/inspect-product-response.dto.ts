export class InspectProductResponseDto {
    constructor(
        public readonly score: number,
        public readonly threshold: number,
        public readonly decision: 'APPROVED' | 'REJECTED',
        public readonly heatmapBase64: string,
        public readonly overlayBase64: string,
    ) { }
}