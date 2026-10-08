import {
    InspectionInferenceGateway,
    InspectionInferenceResult,
} from "../../application/ports/inspection-inference.gateway.js";
import { pythonInspectionResponseSchema } from "./schemas/python-inspection-response.schema.js";

export class PythonInspectionInferenceGateway
    implements InspectionInferenceGateway {

    constructor(
        private readonly inferenceApiUrl: string,
    ) { }


    async inspect(
        image: Buffer,
    ): Promise<InspectionInferenceResult> {

        // Cria o formulário multipart que será enviado ao FastAPI.
        const formData = new FormData();

        const imageBlob = new Blob([
            new Uint8Array(image),
        ]);

        formData.append(
            "image",
            imageBlob,
            "inspection-image.png",
        );


        // Envia a imagem para o serviço Python.
        const response = await fetch(
            `${this.inferenceApiUrl}/inspect`,
            {
                method: "POST",
                body: formData,
            },
        );


        // Trata erros HTTP retornados pelo FastAPI.
        if (!response.ok) {
            const responseBody = await response.text();

            throw new Error(
                `Inference API returned ${response.status}: ${responseBody}`,
            );
        }


        // A resposta HTTP é externa, então inicialmente tratamos como unknown.
        const data: unknown = await response.json();


        // Valida em runtime se o Python retornou a estrutura esperada.
        const parsed =
            pythonInspectionResponseSchema.parse(data);


        // Converte o contrato snake_case do Python
        // para o padrão camelCase usado dentro da aplicação Nest.
        return {
            score: parsed.score,
            threshold: parsed.threshold,
            decision: parsed.decision,

            heatmapBase64: parsed.heatmap_base64,
            overlayBase64: parsed.overlay_base64,
        };
    }
}