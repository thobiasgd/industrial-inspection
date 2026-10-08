import { InspectProductRequestDto } from "../dto/inspect-product-request.dto.js";
import { InspectProductResponseDto } from "../dto/inspect-product-response.dto.js";
import { InspectionInferenceGateway } from "../ports/inspection-inference.gateway.js";

export class InspectProductUseCase {
    constructor(
        private readonly inferenceGateway: InspectionInferenceGateway,
    ) { }

    async execute(
        input: InspectProductRequestDto,
    ): Promise<InspectProductResponseDto> {
        if (input.image.length === 0) {
            throw new Error('Image cannot be empty.');
        }

        // Solicita a inferência através do contrato da aplicação.
        const result = await this.inferenceGateway.inspect(
            input.image,
        );

        return new InspectProductResponseDto(
            result.score,
            result.threshold,
            result.decision,
            result.heatmapBase64,
            result.overlayBase64,
        );
    }
}