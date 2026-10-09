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

            imageWidth: 900,
            imageHeight: 900,

            boundingBoxes: [
                {
                    x: 250,
                    y: 280,
                    width: 360,
                    height: 280,
                },
            ],

            heatmapBase64: "",
            overlayBase64: "",
        };
    }
}