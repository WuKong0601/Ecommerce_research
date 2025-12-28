import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CartService } from '../cart/cart.service';
import { UsersService } from '../users/users.service';
import { CreateOrderDto } from './dto/create-order.dto';

@Injectable()
export class OrdersService {
  constructor(
    private prisma: PrismaService,
    private cartService: CartService,
    private usersService: UsersService,
  ) {}

  async create(userId: string, createOrderDto: CreateOrderDto) {
    const cart = await this.cartService.getCart(userId);

    if (cart.items.length === 0) {
      throw new Error('Cart is empty');
    }

    const contextId = this.getContextId();
    const { timeSlot, isWeekend } = this.getContextInfo();

    const order = await this.prisma.order.create({
      data: {
        userId,
        total: cart.total,
        shippingAddress: createOrderDto.shippingAddress,
        shippingPhone: createOrderDto.shippingPhone,
        contextId,
        timeSlot,
        isWeekend,
        status: 'PENDING',
        items: {
          create: cart.items.map((item) => ({
            productId: item.productId,
            quantity: item.quantity,
            price: item.product.price,
          })),
        },
      },
      include: {
        items: {
          include: {
            product: true,
          },
        },
      },
    });

    await this.cartService.clearCart(userId);

    await this.prisma.userInteraction.create({
      data: {
        userId,
        productId: cart.items[0].productId,
        type: 'PURCHASE',
        contextId,
        timeSlot,
        isWeekend,
      },
    });

    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    await this.usersService.updateSegment(
      userId,
      user.interactionCount + 1,
    );

    return order;
  }

  async findAll(userId: string) {
    return this.prisma.order.findMany({
      where: { userId },
      include: {
        items: {
          include: {
            product: true,
          },
        },
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  async findOne(userId: string, orderId: string) {
    return this.prisma.order.findFirst({
      where: { id: orderId, userId },
      include: {
        items: {
          include: {
            product: true,
          },
        },
      },
    });
  }

  private getContextId(): number {
    const { timeSlot, isWeekend } = this.getContextInfo();
    const timeSlots = ['morning', 'afternoon', 'evening', 'late_night', 'unknown'];
    const timeIndex = timeSlots.indexOf(timeSlot);
    return timeIndex + (isWeekend ? 5 : 0);
  }

  private getContextInfo(): { timeSlot: string; isWeekend: boolean } {
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();

    let timeSlot: string;
    if (hour >= 6 && hour < 12) timeSlot = 'morning';
    else if (hour >= 12 && hour < 18) timeSlot = 'afternoon';
    else if (hour >= 18 && hour < 22) timeSlot = 'evening';
    else if (hour >= 22 || hour < 6) timeSlot = 'late_night';
    else timeSlot = 'unknown';

    const isWeekend = day === 0 || day === 6;

    return { timeSlot, isWeekend };
  }
}
