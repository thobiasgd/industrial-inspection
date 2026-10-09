import {
    InspectionInferenceGateway,
    InspectionInferenceResult,
} from "../../application/ports/inspection-inference.gateway.js";

import {
    pythonInspectionResponseSchema,
} from "./schemas/python-inspection-response.schema.js";


export class PythonInspectionInferenceGateway
    implements InspectionInferenceGateway {

    constructor(
        private readonly inferenceApiUrl: string,
    ) { }


    async inspect(
        image: Buffer,
    ): Promise<InspectionInferenceResult> {

        // Cria o formulário enviado ao serviço Python.
        const formData = new FormData();

        const imageBlob = new Blob([
            new Uint8Array(image),
        ]);

        formData.append(
            "image",
            imageBlob,
            "inspection-image.png",
        );


        // Solicita a inferência.
        const response = await fetch(
            `${this.inferenceApiUrl}/inspect`,
            {
                method: "POST",
                body: formData,
            },
        );


        if (!response.ok) {
            const responseBody = await response.text();

            throw new Error(
                `Inference API returned ${response.status}: ${responseBody}`,
            );
        }


        const data: unknown = await response.json();

        // Valida o contrato recebido do Python.
        const parsed =
            pythonInspectionResponseSchema.parse(data);


        // Adapta snake_case do Python para camelCase da aplicação.
        return {
            score: parsed.score,
            threshold: parsed.threshold,
            decision: parsed.decision,

            imageWidth: parsed.image_width,
            imageHeight: parsed.image_height,

            boundingBoxes: parsed.bounding_boxes,

            heatmapBase64: parsed.heatmap_base64,
            overlayBase64: parsed.overlay_base64,
        };
    }
}