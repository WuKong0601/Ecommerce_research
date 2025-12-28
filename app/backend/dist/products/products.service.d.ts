import { PrismaService } from '../prisma/prisma.service';
import { CreateProductDto } from './dto/create-product.dto';
import { UpdateProductDto } from './dto/update-product.dto';
export declare class ProductsService {
    private prisma;
    constructor(prisma: PrismaService);
    create(createProductDto: CreateProductDto): Promise<{
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
    }>;
    findAll(params?: {
        category?: string;
        minPrice?: number;
        maxPrice?: number;
        search?: string;
        skip?: number;
        take?: number;
    }): Promise<{
        products: {
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
        }[];
        total: number;
        page: number;
        totalPages: number;
    }>;
    findOne(id: string): Promise<{
        reviews: ({
            user: {
                name: string;
            };
        } & {
            id: string;
            createdAt: Date;
            updatedAt: Date;
            userId: string;
            productId: string;
            rating: number;
            comment: string | null;
        })[];
    } & {
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
    }>;
    update(id: string, updateProductDto: UpdateProductDto): Promise<{
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
    }>;
    remove(id: string): Promise<{
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
    }>;
    getCategories(): Promise<string[]>;
}
