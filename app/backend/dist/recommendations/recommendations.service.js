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
        }
        catch (error) {
            console.error('ML recommendation error:', error);
            return this.getPopularProducts(limit);
        }
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
    async getContextAwareRecommendations(userId, limit = 10) {
        return this.getPersonalizedRecommendations(userId, limit);
    }
    async getPopularProducts(limit) {
        return this.prisma.product.findMany({
            where: { isActive: true },
            take: limit,
            orderBy: { createdAt: 'desc' },
        });
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