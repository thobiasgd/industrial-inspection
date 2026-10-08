import { Injectable } from '@nestjs/common';
import { InspectionStatusResponseDto } from '../dto/inspection-status-response.dto.js';


@Injectable()
export class GetInspectionStatusUseCase {
    execute(): InspectionStatusResponseDto {
        return new InspectionStatusResponseDto(
            'ready',
            'Inspection module is running',
        );
    }
}