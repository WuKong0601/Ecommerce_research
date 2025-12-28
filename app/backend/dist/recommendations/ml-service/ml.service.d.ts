import { ConfigService } from '@nestjs/config';
export declare class MLService {
    private configService;
    private modelPath;
    private pythonPath;
    constructor(configService: ConfigService);
    getRecommendations(params: {
        userId: string;
        userSegment: string;
        contextId: number;
        interactionHistory: string[];
        limit: number;
    }): Promise<Array<{
        productId: string;
        score: number;
    }>>;
    private getFallbackRecommendations;
}
