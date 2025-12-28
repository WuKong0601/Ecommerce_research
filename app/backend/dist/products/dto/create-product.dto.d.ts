export declare class CreateProductDto {
    name: string;
    description: string;
    price: number;
    category: string;
    group: string;
    priceBucket: number;
    ratingLevel: number;
    stock: number;
    imageUrl?: string;
    images?: string[];
}
