import { envSchema } from "./env.schema.js";

export function validateEnv(
    config: Record<string, unknown>,
) {
    // Valida e converte as variáveis de ambiente.
    return envSchema.parse(config);
}