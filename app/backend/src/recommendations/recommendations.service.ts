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

    // For COLD_START users, use context-based recommendations
    if (user.segment === 'COLD_START') {
      return this.getContextBasedRecommendations(contextId, limit);
    }
    
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

    // For REGULAR and POWER users, use interaction-based recommendations
    return this.getInteractionBasedRecommendations(user, contextId, limit);
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

  async getContextAwareRecommendations(
    userId: string,
    timeSlot?: string,
    isWeekend?: boolean,
    limit: number = 10,
  ) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      console.log('User not found, returning popular products');
      return this.getPopularProducts(limit);
    }

    // Calculate contextId from timeSlot and isWeekend
    let contextId: number;
    if (timeSlot && isWeekend !== undefined) {
      contextId = this.calculateContextId(timeSlot, isWeekend);
      console.log(`Context-aware request: timeSlot=${timeSlot}, isWeekend=${isWeekend}, contextId=${contextId}`);
    } else {
      contextId = this.getContextId();
      console.log(`Using current context: contextId=${contextId}`);
    }

    console.log(`User segment: ${user.segment}, userId: ${userId}`);

    // For COLD_START users, always use context-based recommendations
    if (user.segment === 'COLD_START') {
      console.log('Using context-based recommendations for COLD_START user');
      const recs = await this.getContextBasedRecommendations(contextId, limit);
      console.log(`Returning ${recs.length} recommendations`);
      return recs;
    }

    // For other users, use interaction-based recommendations
    console.log('Using interaction-based recommendations for REGULAR/POWER user');
    const user_with_interactions = await this.prisma.user.findUnique({
      where: { id: userId },
      include: {
        interactions: {
          orderBy: { timestamp: 'desc' },
          take: 50,
          include: { product: true },
        },
      },
    });
    
    const recs = await this.getInteractionBasedRecommendations(user_with_interactions, contextId, limit);
    console.log(`Returning ${recs.length} recommendations`);
    return recs;
  }

  private async getPopularProducts(limit: number) {
    // Get products with most interactions
    const popularProductIds = await this.prisma.userInteraction.groupBy({
      by: ['productId'],
      _count: { productId: true },
      orderBy: { _count: { productId: 'desc' } },
      take: limit,
    });

    const products = await this.prisma.product.findMany({
      where: {
        id: { in: popularProductIds.map(p => p.productId) },
        isActive: true,
      },
    });

    // Sort by interaction count
    const productMap = new Map(products.map(p => [p.id, p]));
    return popularProductIds
      .map(p => productMap.get(p.productId))
      .filter(p => p !== undefined);
  }

  private async getContextBasedRecommendations(contextId: number, limit: number) {
    console.log(`getContextBasedRecommendations: contextId=${contextId}, limit=${limit}`);
    
    // Get products that are popular in this specific context
    const contextInteractions = await this.prisma.userInteraction.groupBy({
      by: ['productId'],
      where: { contextId },
      _count: { productId: true },
      orderBy: { _count: { productId: 'desc' } },
      take: limit * 2, // Get more to filter
    });

    console.log(`Found ${contextInteractions.length} products with interactions in context ${contextId}`);

    if (contextInteractions.length === 0) {
      console.log('No context-specific interactions, falling back to popular products');
      // Fallback to overall popular products
      return this.getPopularProducts(limit);
    }

    const productIds = contextInteractions.map(i => i.productId);
    console.log(`Fetching ${productIds.length} products from database`);

    const products = await this.prisma.product.findMany({
      where: {
        id: { in: productIds },
        isActive: true,
      },
    });

    console.log(`Retrieved ${products.length} active products`);

    // Sort by context-specific popularity
    const productMap = new Map(products.map(p => [p.id, p]));
    const result = contextInteractions
      .map(i => productMap.get(i.productId))
      .filter(p => p !== undefined)
      .slice(0, limit);
    
    console.log(`Returning ${result.length} recommendations`);
    return result;
  }

  private async getInteractionBasedRecommendations(
    user: any,
    contextId: number,
    limit: number,
  ) {
    // Get user's interaction history
    const userProductIds = user.interactions.map(i => i.productId);

    if (userProductIds.length === 0) {
      return this.getContextBasedRecommendations(contextId, limit);
    }

    // Find similar users who interacted with same products
    const similarUserInteractions = await this.prisma.userInteraction.findMany({
      where: {
        productId: { in: userProductIds },
        userId: { not: user.id },
        contextId, // Same context
      },
      select: { userId: true, productId: true },
    });

    // Count product frequencies from similar users
    const productScores = new Map<string, number>();
    similarUserInteractions.forEach(interaction => {
      if (!userProductIds.includes(interaction.productId)) {
        const current = productScores.get(interaction.productId) || 0;
        productScores.set(interaction.productId, current + 1);
      }
    });

    // Sort by score
    const sortedProducts = Array.from(productScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([productId]) => productId);

    if (sortedProducts.length === 0) {
      return this.getContextBasedRecommendations(contextId, limit);
    }

    const products = await this.prisma.product.findMany({
      where: {
        id: { in: sortedProducts },
        isActive: true,
      },
    });

    // Maintain sort order
    const productMap = new Map(products.map(p => [p.id, p]));
    return sortedProducts
      .map(id => productMap.get(id))
      .filter(p => p !== undefined);
  }

  private calculateContextId(timeSlot: string, isWeekend: boolean): number {
    const timeSlotMap = {
      morning: 0,
      afternoon: 2,
      evening: 4,
      late_night: 6,
      unknown: 8,
    };

    const baseId = timeSlotMap[timeSlot] || 8;
    return baseId + (isWeekend ? 1 : 0);
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
