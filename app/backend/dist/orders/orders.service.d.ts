import { PrismaService } from '../prisma/prisma.service';
import { CartService } from '../cart/cart.service';
import { UsersService } from '../users/users.service';
import { CreateOrderDto } from './dto/create-order.dto';
export declare class OrdersService {
    private prisma;
    private cartService;
    private usersService;
    constructor(prisma: PrismaService, cartService: CartService, usersService: UsersService);
    create(userId: string, createOrderDto: CreateOrderDto): Promise<{
        items: ({
            product: {
                name: string;
                description: string;
                id: string;
                createdAt: Date;
                updatedAt: Date;
                price: import("@prisma/client/runtime/library").Decimal;
                category: string;
                group: string;
                priceBucket: number;
                ratingLevel: number;
                stock: number;
                imageUrl: string | null;
                images: string[];
                isActive: boolean;
            };
        } & {
            id: string;
            price: import("@prisma/client/runtime/library").Decimal;
            productId: string;
            quantity: number;
            orderId: string;
        })[];
    } & {
        id: string;
        createdAt: Date;
        updatedAt: Date;
        total: import("@prisma/client/runtime/library").Decimal;
        userId: string;
        shippingAddress: string;
        shippingPhone: string;
        timeSlot: string | null;
        isWeekend: boolean | null;
        status: import(".prisma/client").$Enums.OrderStatus;
        contextId: number | null;
    }>;
    findAll(userId: string): Promise<({
        items: ({
            product: {
                name: string;
                description: string;
                id: string;
                createdAt: Date;
                updatedAt: Date;
                price: import("@prisma/client/runtime/library").Decimal;
                category: string;
                group: string;
                priceBucket: number;
                ratingLevel: number;
                stock: number;
                imageUrl: string | null;
                images: string[];
                isActive: boolean;
            };
        } & {
            id: string;
            price: import("@prisma/client/runtime/library").Decimal;
            productId: string;
            quantity: number;
            orderId: string;
        })[];
    } & {
        id: string;
        createdAt: Date;
        updatedAt: Date;
        total: import("@prisma/client/runtime/library").Decimal;
        userId: string;
        shippingAddress: string;
        shippingPhone: string;
        timeSlot: string | null;
        isWeekend: boolean | null;
        status: import(".prisma/client").$Enums.OrderStatus;
        contextId: number | null;
    })[]>;
    findOne(userId: string, orderId: string): Promise<{
        items: ({
            product: {
                name: string;
                description: string;
                id: string;
                createdAt: Date;
                updatedAt: Date;
                price: import("@prisma/client/runtime/library").Decimal;
                category: string;
                group: string;
                priceBucket: number;
                ratingLevel: number;
                stock: number;
                imageUrl: string | null;
                images: string[];
                isActive: boolean;
            };
        } & {
            id: string;
            price: import("@prisma/client/runtime/library").Decimal;
            productId: string;
            quantity: number;
            orderId: string;
        })[];
    } & {
        id: string;
        createdAt: Date;
        updatedAt: Date;
        total: import("@prisma/client/runtime/library").Decimal;
        userId: string;
        shippingAddress: string;
        shippingPhone: string;
        timeSlot: string | null;
        isWeekend: boolean | null;
        status: import(".prisma/client").$Enums.OrderStatus;
        contextId: number | null;
    }>;
    private getContextId;
    private getContextInfo;
}
