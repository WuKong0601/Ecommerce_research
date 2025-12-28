import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class UsersService {
  constructor(private prisma: PrismaService) {}

  async findByEmail(email: string) {
    return this.prisma.user.findUnique({ where: { email } });
  }

  async findById(id: string) {
    return this.prisma.user.findUnique({
      where: { id },
      select: {
        id: true,
        email: true,
        name: true,
        phone: true,
        role: true,
        segment: true,
        interactionCount: true,
        createdAt: true,
      },
    });
  }

  async updateSegment(userId: string, interactionCount: number) {
    let segment: 'COLD_START' | 'REGULAR' | 'POWER';
    
    if (interactionCount === 1) {
      segment = 'COLD_START';
    } else if (interactionCount >= 2 && interactionCount <= 4) {
      segment = 'REGULAR';
    } else {
      segment = 'POWER';
    }

    return this.prisma.user.update({
      where: { id: userId },
      data: { segment, interactionCount },
    });
  }
}
