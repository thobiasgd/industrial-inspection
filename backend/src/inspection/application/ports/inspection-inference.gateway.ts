export interface InspectionBoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}


export interface InspectionInferenceResult {
    score: number;
    threshold: number;

    decision:
    | "APPROVED"
    | "REJECTED";

    imageWidth: number;
    imageHeight: number;

    boundingBoxes: InspectionBoundingBox[];

    heatmapBase64: string;
    overlayBase64: string;
}


export interface InspectionInferenceGateway {
    inspect(
        image: Buffer,
    ): Promise<InspectionInferenceResult>;
}