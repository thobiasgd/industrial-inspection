import type {
    InspectionBoundingBox,
} from "../ports/inspection-inference.gateway.js";


export class InspectProductResponseDto {
    constructor(
        public readonly score: number,

        public readonly threshold: number,

        public readonly decision:
            | "APPROVED"
            | "REJECTED",

        public readonly imageWidth: number,

        public readonly imageHeight: number,

        public readonly boundingBoxes:
            InspectionBoundingBox[],

        public readonly heatmapBase64: string,

        public readonly overlayBase64: string,
    ) { }
}