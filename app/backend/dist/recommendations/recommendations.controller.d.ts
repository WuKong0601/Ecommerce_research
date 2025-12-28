import { RecommendationsService } from './recommendations.service';
export declare class RecommendationsController {
    private recommendationsService;
    constructor(recommendationsService: RecommendationsService);
    getPersonalized(req: any, limit?: string): Promise<{
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
    }[]>;
    getContextAware(req: any, limit?: string): Promise<{
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
    }[]>;
    getSimilar(productId: string, limit?: string): Promise<{
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
    }[]>;
}
