import {
    InspectionInferenceGateway,
    InspectionInferenceResult,
} from "../../application/ports/inspection-inference.gateway.js";


export class FakeInspectionInferenceGateway
    implements InspectionInferenceGateway {

    async inspect(
        _image: Buffer,
    ): Promise<InspectionInferenceResult> {

        // Retorna dados fixos para testes sem depender do serviço Python.
        return {
            score: 2.6701,
            threshold: 2.2029,
            decision: "REJECTED",

            heatmapBase64: "",
            overlayBase64: "",
        };
    }
}