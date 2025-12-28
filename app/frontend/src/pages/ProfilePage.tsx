import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'
import { useAuthStore } from '@/store/auth'
import { User, Mail, Phone, Award } from 'lucide-react'

export function ProfilePage() {
  const { user } = useAuthStore()

  const { data: profile } = useQuery({
    queryKey: ['profile'],
    queryFn: async () => {
      const { data } = await apiClient.get('/users/profile')
      return data
    },
  })

  const getSegmentBadge = (segment: string) => {
    const badges = {
      COLD_START: { label: 'New User', color: 'bg-blue-100 text-blue-800' },
      REGULAR: { label: 'Regular Shopper', color: 'bg-green-100 text-green-800' },
      POWER: { label: 'Power User', color: 'bg-purple-100 text-purple-800' },
    }
    return badges[segment as keyof typeof badges] || badges.COLD_START
  }

  const badge = getSegmentBadge(profile?.segment || user?.segment || 'COLD_START')

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">My Profile</h1>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Personal Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <User className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Name</p>
                <p className="font-medium">{profile?.name || user?.name}</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Mail className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Email</p>
                <p className="font-medium">{profile?.email || user?.email}</p>
              </div>
            </div>

            {profile?.phone && (
              <div className="flex items-center gap-3">
                <Phone className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Phone</p>
                  <p className="font-medium">{profile.phone}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Shopping Profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <Award className="h-5 w-5 text-muted-foreground" />
              <div className="flex-1">
                <p className="text-sm text-muted-foreground">User Segment</p>
                <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${badge.color}`}>
                  {badge.label}
                </span>
              </div>
            </div>

            <div>
              <p className="text-sm text-muted-foreground">Total Interactions</p>
              <p className="text-2xl font-bold">{profile?.interactionCount || 0}</p>
            </div>

            <div className="bg-muted/50 p-4 rounded-lg">
              <p className="text-sm font-medium mb-2">About User Segments:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• <strong>New User:</strong> 1 interaction - Getting started</li>
                <li>• <strong>Regular Shopper:</strong> 2-4 interactions - Active user</li>
                <li>• <strong>Power User:</strong> 5+ interactions - VIP experience</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
