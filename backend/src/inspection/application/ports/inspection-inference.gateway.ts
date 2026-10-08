export interface InspectionInferenceResult {
    score: number;
    threshold: number;

    decision:
    | "APPROVED"
    | "REJECTED";

    heatmapBase64: string;
    overlayBase64: string;
}


export interface InspectionInferenceGateway {
    inspect(
        image: Buffer,
    ): Promise<InspectionInferenceResult>;
}