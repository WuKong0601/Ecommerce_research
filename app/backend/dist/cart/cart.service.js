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
exports.CartService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
let CartService = class CartService {
    constructor(prisma) {
        this.prisma = prisma;
    }
    async getCart(userId) {
        const items = await this.prisma.cartItem.findMany({
            where: { userId },
            include: {
                product: true,
            },
        });
        const total = items.reduce((sum, item) => {
            return sum + Number(item.product.price) * item.quantity;
        }, 0);
        return { items, total };
    }
    async addItem(userId, productId, quantity = 1) {
        const existing = await this.prisma.cartItem.findUnique({
            where: {
                userId_productId: { userId, productId },
            },
        });
        if (existing) {
            return this.prisma.cartItem.update({
                where: { id: existing.id },
                data: { quantity: existing.quantity + quantity },
                include: { product: true },
            });
        }
        return this.prisma.cartItem.create({
            data: { userId, productId, quantity },
            include: { product: true },
        });
    }
    async updateItem(userId, itemId, quantity) {
        return this.prisma.cartItem.update({
            where: { id: itemId, userId },
            data: { quantity },
            include: { product: true },
        });
    }
    async removeItem(userId, itemId) {
        return this.prisma.cartItem.delete({
            where: { id: itemId, userId },
        });
    }
    async clearCart(userId) {
        return this.prisma.cartItem.deleteMany({
            where: { userId },
        });
    }
};
exports.CartService = CartService;
exports.CartService = CartService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], CartService);
//# sourceMappingURL=cart.service.js.map