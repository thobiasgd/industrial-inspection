import { z } from "zod";


const boundingBoxSchema = z.object({
    x: z.number().int().nonnegative(),
    y: z.number().int().nonnegative(),

    width: z.number().int().positive(),
    height: z.number().int().positive(),
});


export const pythonInspectionResponseSchema = z.object({
    score: z.number(),

    threshold: z.number(),

    decision: z.enum([
        "APPROVED",
        "REJECTED",
    ]),

    image_width: z.number()
        .int()
        .positive(),

    image_height: z.number()
        .int()
        .positive(),

    bounding_boxes: z.array(
        boundingBoxSchema,
    ),

    heatmap_base64: z.string(),

    overlay_base64: z.string(),
});


export type PythonInspectionResponse = z.infer<
    typeof pythonInspectionResponseSchema
>;