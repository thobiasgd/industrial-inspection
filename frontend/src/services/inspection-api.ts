import { z } from 'zod';


const inspectionStatusSchema = z.object({
    status: z.string(),
    message: z.string(),
});


const inspectionResultSchema = z.object({
    score: z.number(),
    threshold: z.number(),

    decision: z.enum([
        'APPROVED',
        'REJECTED',
    ]),

    heatmapBase64: z.string(),
    overlayBase64: z.string(),
});


export type InspectionStatus = z.infer<
    typeof inspectionStatusSchema
>;

export type InspectionResult = z.infer<
    typeof inspectionResultSchema
>;


const apiUrl = import.meta.env.VITE_API_URL;


export async function getInspectionStatus(): Promise<InspectionStatus> {
    const response = await fetch(
        `${apiUrl}/inspections/status`,
    );

    if (!response.ok) {
        throw new Error(
            `API returned status ${response.status}`,
        );
    }

    const data: unknown = await response.json();

    return inspectionStatusSchema.parse(data);
}


export async function inspectProduct(
    image: File,
): Promise<InspectionResult> {

    // Monta o multipart/form-data que o Nest espera.
    const formData = new FormData();

    formData.append(
        'image',
        image,
    );

    const response = await fetch(
        `${apiUrl}/inspections`,
        {
            method: 'POST',
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            `Inspection failed with status ${response.status}`,
        );
    }

    const data: unknown = await response.json();

    // Valida em runtime a resposta recebida do backend.
    return inspectionResultSchema.parse(data);
}