import { Module } from '@nestjs/common';
import { RecommendationsService } from './recommendations.service';
import { RecommendationsController } from './recommendations.controller';
import { MLService } from './ml-service/ml.service';

@Module({
  controllers: [RecommendationsController],
  providers: [RecommendationsService, MLService],
})
export class RecommendationsModule {}
