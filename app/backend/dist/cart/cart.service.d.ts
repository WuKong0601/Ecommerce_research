import { PrismaService } from '../prisma/prisma.service';
export declare class CartService {
    private prisma;
    constructor(prisma: PrismaService);
    getCart(userId: string): Promise<{
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
            createdAt: Date;
            updatedAt: Date;
            userId: string;
            productId: string;
            quantity: number;
        })[];
        total: number;
    }>;
    addItem(userId: string, productId: string, quantity?: number): Promise<{
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
        createdAt: Date;
        updatedAt: Date;
        userId: string;
        productId: string;
        quantity: number;
    }>;
    updateItem(userId: string, itemId: string, quantity: number): Promise<{
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
        createdAt: Date;
        updatedAt: Date;
        userId: string;
        productId: string;
        quantity: number;
    }>;
    removeItem(userId: string, itemId: string): Promise<{
        id: string;
        createdAt: Date;
        updatedAt: Date;
        userId: string;
        productId: string;
        quantity: number;
    }>;
    clearCart(userId: string): Promise<import(".prisma/client").Prisma.BatchPayload>;
}
