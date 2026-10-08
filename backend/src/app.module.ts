import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { validateEnv } from './config/env.validation.js';
import { InspectionModule } from './inspection/inspection.module.js';



@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,

      // Valida as variáveis antes de iniciar a aplicação.
      validate: validateEnv,
    }),

    InspectionModule,
  ],
})
export class AppModule { }