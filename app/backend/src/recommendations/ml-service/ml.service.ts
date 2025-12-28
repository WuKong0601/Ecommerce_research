import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as path from 'path';
import * as fs from 'fs';

@Injectable()
export class MLService {
  private modelPath: string;
  private pythonPath: string;

  constructor(private configService: ConfigService) {
    this.modelPath = this.configService.get('ML_MODEL_PATH') || '../../results/models/best_model.pt';
    this.pythonPath = this.configService.get('PYTHON_PATH') || 'python';
  }

  async getRecommendations(params: {
    userId: string;
    userSegment: string;
    contextId: number;
    interactionHistory: string[];
    limit: number;
  }): Promise<Array<{ productId: string; score: number }>> {
    if (!fs.existsSync(this.modelPath)) {
      console.warn('ML model not found, using fallback recommendations');
      return this.getFallbackRecommendations(params);
    }

    return this.getFallbackRecommendations(params);
  }

  private async getFallbackRecommendations(params: {
    userId: string;
    userSegment: string;
    contextId: number;
    interactionHistory: string[];
    limit: number;
  }): Promise<Array<{ productId: string; score: number }>> {
    return [];
  }
}
