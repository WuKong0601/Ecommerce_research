import { JwtService } from '@nestjs/jwt';
import { PrismaService } from '../prisma/prisma.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
export declare class AuthService {
    private prisma;
    private jwtService;
    constructor(prisma: PrismaService, jwtService: JwtService);
    register(registerDto: RegisterDto): Promise<{
        accessToken: string;
        user: {
            name: string;
            email: string;
            phone: string | null;
            id: string;
            role: import(".prisma/client").$Enums.UserRole;
            segment: import(".prisma/client").$Enums.UserSegment;
            interactionCount: number;
            createdAt: Date;
            updatedAt: Date;
        };
    }>;
    login(loginDto: LoginDto): Promise<{
        accessToken: string;
        user: {
            name: string;
            email: string;
            phone: string | null;
            id: string;
            role: import(".prisma/client").$Enums.UserRole;
            segment: import(".prisma/client").$Enums.UserSegment;
            interactionCount: number;
            createdAt: Date;
            updatedAt: Date;
        };
    }>;
    validateUser(email: string, password: string): Promise<{
        name: string;
        email: string;
        phone: string | null;
        id: string;
        role: import(".prisma/client").$Enums.UserRole;
        segment: import(".prisma/client").$Enums.UserSegment;
        interactionCount: number;
        createdAt: Date;
        updatedAt: Date;
    }>;
    generateTokens(userId: string, email: string): Promise<{
        accessToken: string;
    }>;
    getUserById(userId: string): Promise<{
        name: string;
        email: string;
        phone: string;
        id: string;
        role: import(".prisma/client").$Enums.UserRole;
        segment: import(".prisma/client").$Enums.UserSegment;
        interactionCount: number;
        createdAt: Date;
    }>;
}
