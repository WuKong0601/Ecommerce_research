"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RecommendationsService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const ml_service_1 = require("./ml-service/ml.service");
let RecommendationsService = class RecommendationsService {
    constructor(prisma, mlService) {
        this.prisma = prisma;
        this.mlService = mlService;
    }
    async getPersonalizedRecommendations(userId, limit = 10) {
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
        return this.getInteractionBasedRecommendations(user, contextId, limit);
    }
    async getSimilarProducts(productId, limit = 6) {
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
    async getContextAwareRecommendations(userId, timeSlot, isWeekend, limit = 10) {
        const user = await this.prisma.user.findUnique({
            where: { id: userId },
        });
        if (!user) {
            console.log('User not found, returning popular products');
            return this.getPopularProducts(limit);
        }
        let contextId;
        if (timeSlot && isWeekend !== undefined) {
            contextId = this.calculateContextId(timeSlot, isWeekend);
            console.log(`Context-aware request: timeSlot=${timeSlot}, isWeekend=${isWeekend}, contextId=${contextId}`);
        }
        else {
            contextId = this.getContextId();
            console.log(`Using current context: contextId=${contextId}`);
        }
        console.log(`User segment: ${user.segment}, userId: ${userId}`);
        if (user.segment === 'COLD_START') {
            console.log('Using context-based recommendations for COLD_START user');
            const recs = await this.getContextBasedRecommendations(contextId, limit);
            console.log(`Returning ${recs.length} recommendations`);
            return recs;
        }
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
    async getPopularProducts(limit) {
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
        const productMap = new Map(products.map(p => [p.id, p]));
        return popularProductIds
            .map(p => productMap.get(p.productId))
            .filter(p => p !== undefined);
    }
    async getContextBasedRecommendations(contextId, limit) {
        console.log(`getContextBasedRecommendations: contextId=${contextId}, limit=${limit}`);
        const contextInteractions = await this.prisma.userInteraction.groupBy({
            by: ['productId'],
            where: { contextId },
            _count: { productId: true },
            orderBy: { _count: { productId: 'desc' } },
            take: limit * 2,
        });
        console.log(`Found ${contextInteractions.length} products with interactions in context ${contextId}`);
        if (contextInteractions.length === 0) {
            console.log('No context-specific interactions, falling back to popular products');
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
        const productMap = new Map(products.map(p => [p.id, p]));
        const result = contextInteractions
            .map(i => productMap.get(i.productId))
            .filter(p => p !== undefined)
            .slice(0, limit);
        console.log(`Returning ${result.length} recommendations`);
        return result;
    }
    async getInteractionBasedRecommendations(user, contextId, limit) {
        const userProductIds = user.interactions.map(i => i.productId);
        if (userProductIds.length === 0) {
            return this.getContextBasedRecommendations(contextId, limit);
        }
        const similarUserInteractions = await this.prisma.userInteraction.findMany({
            where: {
                productId: { in: userProductIds },
                userId: { not: user.id },
                contextId,
            },
            select: { userId: true, productId: true },
        });
        const productScores = new Map();
        similarUserInteractions.forEach(interaction => {
            if (!userProductIds.includes(interaction.productId)) {
                const current = productScores.get(interaction.productId) || 0;
                productScores.set(interaction.productId, current + 1);
            }
        });
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
        const productMap = new Map(products.map(p => [p.id, p]));
        return sortedProducts
            .map(id => productMap.get(id))
            .filter(p => p !== undefined);
    }
    calculateContextId(timeSlot, isWeekend) {
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
    getContextId() {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();
        let timeSlot;
        if (hour >= 6 && hour < 12)
            timeSlot = 0;
        else if (hour >= 12 && hour < 18)
            timeSlot = 1;
        else if (hour >= 18 && hour < 22)
            timeSlot = 2;
        else if (hour >= 22 || hour < 6)
            timeSlot = 3;
        else
            timeSlot = 4;
        const isWeekend = day === 0 || day === 6;
        return timeSlot + (isWeekend ? 5 : 0);
    }
};
exports.RecommendationsService = RecommendationsService;
exports.RecommendationsService = RecommendationsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService,
        ml_service_1.MLService])
], RecommendationsService);
//# sourceMappingURL=recommendations.service.js.map