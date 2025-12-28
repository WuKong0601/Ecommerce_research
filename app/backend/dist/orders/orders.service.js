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
exports.OrdersService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const cart_service_1 = require("../cart/cart.service");
const users_service_1 = require("../users/users.service");
let OrdersService = class OrdersService {
    constructor(prisma, cartService, usersService) {
        this.prisma = prisma;
        this.cartService = cartService;
        this.usersService = usersService;
    }
    async create(userId, createOrderDto) {
        const cart = await this.cartService.getCart(userId);
        if (cart.items.length === 0) {
            throw new Error('Cart is empty');
        }
        const contextId = this.getContextId();
        const { timeSlot, isWeekend } = this.getContextInfo();
        const order = await this.prisma.order.create({
            data: {
                userId,
                total: cart.total,
                shippingAddress: createOrderDto.shippingAddress,
                shippingPhone: createOrderDto.shippingPhone,
                contextId,
                timeSlot,
                isWeekend,
                status: 'PENDING',
                items: {
                    create: cart.items.map((item) => ({
                        productId: item.productId,
                        quantity: item.quantity,
                        price: item.product.price,
                    })),
                },
            },
            include: {
                items: {
                    include: {
                        product: true,
                    },
                },
            },
        });
        await this.cartService.clearCart(userId);
        await this.prisma.userInteraction.create({
            data: {
                userId,
                productId: cart.items[0].productId,
                type: 'PURCHASE',
                contextId,
                timeSlot,
                isWeekend,
            },
        });
        const user = await this.prisma.user.findUnique({
            where: { id: userId },
        });
        await this.usersService.updateSegment(userId, user.interactionCount + 1);
        return order;
    }
    async findAll(userId) {
        return this.prisma.order.findMany({
            where: { userId },
            include: {
                items: {
                    include: {
                        product: true,
                    },
                },
            },
            orderBy: { createdAt: 'desc' },
        });
    }
    async findOne(userId, orderId) {
        return this.prisma.order.findFirst({
            where: { id: orderId, userId },
            include: {
                items: {
                    include: {
                        product: true,
                    },
                },
            },
        });
    }
    getContextId() {
        const { timeSlot, isWeekend } = this.getContextInfo();
        const timeSlots = ['morning', 'afternoon', 'evening', 'late_night', 'unknown'];
        const timeIndex = timeSlots.indexOf(timeSlot);
        return timeIndex + (isWeekend ? 5 : 0);
    }
    getContextInfo() {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();
        let timeSlot;
        if (hour >= 6 && hour < 12)
            timeSlot = 'morning';
        else if (hour >= 12 && hour < 18)
            timeSlot = 'afternoon';
        else if (hour >= 18 && hour < 22)
            timeSlot = 'evening';
        else if (hour >= 22 || hour < 6)
            timeSlot = 'late_night';
        else
            timeSlot = 'unknown';
        const isWeekend = day === 0 || day === 6;
        return { timeSlot, isWeekend };
    }
};
exports.OrdersService = OrdersService;
exports.OrdersService = OrdersService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService,
        cart_service_1.CartService,
        users_service_1.UsersService])
], OrdersService);
//# sourceMappingURL=orders.service.js.map