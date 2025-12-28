import {
  Controller,
  Get,
  Param,
  Query,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiBearerAuth } from '@nestjs/swagger';
import { RecommendationsService } from './recommendations.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('recommendations')
@Controller('recommendations')
export class RecommendationsController {
  constructor(private recommendationsService: RecommendationsService) {}

  @Get('for-you')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  getPersonalized(
    @Request() req,
    @Query('limit') limit?: string,
  ) {
    return this.recommendationsService.getPersonalizedRecommendations(
      req.user.userId,
      limit ? parseInt(limit) : 10,
    );
  }

  @Get('context-aware')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  getContextAware(
    @Request() req,
    @Query('timeSlot') timeSlot?: string,
    @Query('isWeekend') isWeekend?: string,
    @Query('limit') limit?: string,
  ) {
    return this.recommendationsService.getContextAwareRecommendations(
      req.user.userId,
      timeSlot,
      isWeekend === 'true',
      limit ? parseInt(limit) : 10,
    );
  }

  @Get('similar/:productId')
  getSimilar(
    @Param('productId') productId: string,
    @Query('limit') limit?: string,
  ) {
    return this.recommendationsService.getSimilarProducts(
      productId,
      limit ? parseInt(limit) : 6,
    );
  }
}
