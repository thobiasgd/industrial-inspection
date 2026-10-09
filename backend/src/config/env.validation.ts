import { envSchema } from "./env.schema.js";

export function validateEnv(
    config: Record<string, unknown>,
) {
    return envSchema.parse(config);
}