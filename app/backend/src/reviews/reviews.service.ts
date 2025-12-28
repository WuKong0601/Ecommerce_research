import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateReviewDto } from './dto/create-review.dto';

@Injectable()
export class ReviewsService {
  constructor(private prisma: PrismaService) {}

  async create(userId: string, createReviewDto: CreateReviewDto) {
    const review = await this.prisma.review.create({
      data: {
        userId,
        ...createReviewDto,
      },
      include: {
        user: {
          select: { name: true },
        },
        product: true,
      },
    });

    await this.prisma.userInteraction.create({
      data: {
        userId,
        productId: createReviewDto.productId,
        type: 'REVIEW',
      },
    });

    return review;
  }

  async findByProduct(productId: string) {
    return this.prisma.review.findMany({
      where: { productId },
      include: {
        user: {
          select: { name: true },
        },
      },
      orderBy: { createdAt: 'desc' },
    });
  }
}
