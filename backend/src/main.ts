import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';

import { AppModule } from './app.module.js';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  const configService = app.get(ConfigService);

  const port = configService.getOrThrow<number>(
    'PORT',
  );

  app.enableCors({
    origin: 'http://localhost:5173',
  });

  await app.listen(port);
}

void bootstrap();