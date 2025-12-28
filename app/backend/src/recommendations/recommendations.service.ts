import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { MLService } from './ml-service/ml.service';

@Injectable()
export class RecommendationsService {
  constructor(
    private prisma: PrismaService,
    private mlService: MLService,
  ) {}

  async getPersonalizedRecommendations(userId: string, limit: number = 10) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      include: {
        interactions: {
          orderBy: { timestamp: 'desc' },
          take: 50,
          include: { product: true },
        },
      },
    });

    if (!user) {
      return this.getPopularProducts(limit);
    }

    const contextId = this.getContextId();
    
    const cachedRecs = await this.prisma.recommendationCache.findUnique({
      where: {
        userId_contextId: { userId, contextId },
      },
    });

    if (cachedRecs && cachedRecs.expiresAt > new Date()) {
      const products = await this.prisma.product.findMany({
        where: {
          id: { in: cachedRecs.productIds },
          isActive: true,
        },
      });

      return products.slice(0, limit);
    }

    try {
      const recommendations = await this.mlService.getRecommendations({
        userId,
        userSegment: user.segment,
        contextId,
        interactionHistory: user.interactions.map(i => i.productId),
        limit,
      });

      await this.prisma.recommendationCache.upsert({
        where: {
          userId_contextId: { userId, contextId },
        },
        create: {
          userId,
          contextId,
          productIds: recommendations.map(r => r.productId),
          scores: recommendations.map(r => r.score),
          expiresAt: new Date(Date.now() + 3600000),
        },
        update: {
          productIds: recommendations.map(r => r.productId),
          scores: recommendations.map(r => r.score),
          expiresAt: new Date(Date.now() + 3600000),
        },
      });

      const products = await this.prisma.product.findMany({
        where: {
          id: { in: recommendations.map(r => r.productId) },
          isActive: true,
        },
      });

      return products;
    } catch (error) {
      console.error('ML recommendation error:', error);
      return this.getPopularProducts(limit);
    }
  }

  async getSimilarProducts(productId: string, limit: number = 6) {
    const product = await this.prisma.product.findUnique({
      where: { id: productId },
    });

    if (!product) {
      return [];
    }

    return this.prisma.product.findMany({
      where: {
        category: product.category,
        id: { not: productId },
        isActive: true,
      },
      take: limit,
      orderBy: { createdAt: 'desc' },
    });
  }

  async getContextAwareRecommendations(userId: string, limit: number = 10) {
    return this.getPersonalizedRecommendations(userId, limit);
  }

  private async getPopularProducts(limit: number) {
    return this.prisma.product.findMany({
      where: { isActive: true },
      take: limit,
      orderBy: { createdAt: 'desc' },
    });
  }

  private getContextId(): number {
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();

    let timeSlot: number;
    if (hour >= 6 && hour < 12) timeSlot = 0;
    else if (hour >= 12 && hour < 18) timeSlot = 1;
    else if (hour >= 18 && hour < 22) timeSlot = 2;
    else if (hour >= 22 || hour < 6) timeSlot = 3;
    else timeSlot = 4;

    const isWeekend = day === 0 || day === 6;
    return timeSlot + (isWeekend ? 5 : 0);
  }
}
