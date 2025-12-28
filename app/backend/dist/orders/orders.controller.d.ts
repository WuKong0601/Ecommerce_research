import { OrdersService } from './orders.service';
import { CreateOrderDto } from './dto/create-order.dto';
export declare class OrdersController {
    private ordersService;
    constructor(ordersService: OrdersService);
    create(req: any, createOrderDto: CreateOrderDto): Promise<{
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
    findAll(req: any): Promise<({
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
    findOne(req: any, id: string): Promise<{
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
}
