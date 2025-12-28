import { AuthService } from './auth.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
export declare class AuthController {
    private authService;
    constructor(authService: AuthService);
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
    getProfile(req: any): Promise<{
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
