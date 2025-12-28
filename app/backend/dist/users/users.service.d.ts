import { PrismaService } from '../prisma/prisma.service';
export declare class UsersService {
    private prisma;
    constructor(prisma: PrismaService);
    findByEmail(email: string): Promise<{
        name: string;
        email: string;
        password: string;
        phone: string | null;
        id: string;
        role: import(".prisma/client").$Enums.UserRole;
        segment: import(".prisma/client").$Enums.UserSegment;
        interactionCount: number;
        createdAt: Date;
        updatedAt: Date;
    }>;
    findById(id: string): Promise<{
        name: string;
        email: string;
        phone: string;
        id: string;
        role: import(".prisma/client").$Enums.UserRole;
        segment: import(".prisma/client").$Enums.UserSegment;
        interactionCount: number;
        createdAt: Date;
    }>;
    updateSegment(userId: string, interactionCount: number): Promise<{
        name: string;
        email: string;
        password: string;
        phone: string | null;
        id: string;
        role: import(".prisma/client").$Enums.UserRole;
        segment: import(".prisma/client").$Enums.UserSegment;
        interactionCount: number;
        createdAt: Date;
        updatedAt: Date;
    }>;
}
