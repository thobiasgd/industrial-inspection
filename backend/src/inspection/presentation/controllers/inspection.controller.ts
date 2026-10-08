/// <reference types="multer" />
import { BadRequestException, Controller, Get, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { InspectionStatusResponseDto } from '../../application/dto/inspection-status-response.dto.js';
import { GetInspectionStatusUseCase } from '../../application/use-cases/get-inspection-status.use-case.js';
import { InspectProductUseCase } from '../../application/use-cases/inspect-product.use-case.js';
import { InspectProductResponseDto } from '../../application/dto/inspect-product-response.dto.js';
import { InspectProductRequestDto } from '../../application/dto/inspect-product-request.dto.js';
import { FileInterceptor } from '@nestjs/platform-express';

@Controller('inspections')
export class InspectionController {
    constructor(
        private readonly getInspectionStatusUseCase: GetInspectionStatusUseCase,
        private readonly inspectProductUseCase: InspectProductUseCase,
    ) { }


    @Get('status')
    getStatus(): InspectionStatusResponseDto {
        return this.getInspectionStatusUseCase.execute();
    }


    @Post()
    @UseInterceptors(
        FileInterceptor('image'),
    )
    async inspect(
        @UploadedFile()
        file: Express.Multer.File | undefined,
    ): Promise<InspectProductResponseDto> {

        if (!file) {
            throw new BadRequestException(
                'Image is required.',
            );
        }

        const input = new InspectProductRequestDto(
            file.buffer,
        );

        return this.inspectProductUseCase.execute(
            input,
        );
    }
}