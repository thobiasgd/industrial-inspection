import { z } from "zod";


export const pythonInspectionResponseSchema = z.object({
    score: z.number(),

    threshold: z.number(),

    decision: z.enum([
        "APPROVED",
        "REJECTED",
    ]),

    heatmap_base64: z.string(),

    overlay_base64: z.string(),
});


export type PythonInspectionResponse = z.infer<
    typeof pythonInspectionResponseSchema
>;