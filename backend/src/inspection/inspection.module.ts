import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InspectionController } from './presentation/controllers/inspection.controller.js';
import { GetInspectionStatusUseCase } from './application/use-cases/get-inspection-status.use-case.js';
import { INSPECTION_INFERENCE_GATEWAY } from './application/ports/inspection-inference.token.js';
import { PythonInspectionInferenceGateway } from './infrastructure/inference/python-inspection-inference.gateway.js';
import { InspectionInferenceGateway } from './application/ports/inspection-inference.gateway.js';
import { InspectProductUseCase } from './application/use-cases/inspect-product.use-case.js';


@Module({
    controllers: [
        InspectionController,
    ],

    providers: [
        GetInspectionStatusUseCase,

        {
            provide: INSPECTION_INFERENCE_GATEWAY,
            inject: [ConfigService],

            useFactory: (
                configService: ConfigService,
            ) => {
                const inferenceApiUrl =
                    configService.getOrThrow<string>(
                        'INFERENCE_API_URL',
                    );

                return new PythonInspectionInferenceGateway(
                    inferenceApiUrl,
                );
            },
        },

        {
            provide: InspectProductUseCase,

            inject: [
                INSPECTION_INFERENCE_GATEWAY,
            ],

            useFactory: (
                inferenceGateway: InspectionInferenceGateway,
            ) => {
                return new InspectProductUseCase(
                    inferenceGateway,
                );
            },
        },
    ],
})
export class InspectionModule { }