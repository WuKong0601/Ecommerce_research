import { UsersService } from './users.service';
export declare class UsersController {
    private usersService;
    constructor(usersService: UsersService);
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
