import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Clock, Calendar, Sun, Moon, Sunset, CloudMoon } from 'lucide-react'

interface ContextSelectorProps {
  onContextChange: (timeSlot: string, isWeekend: boolean) => void
  currentTimeSlot?: string
  currentIsWeekend?: boolean
}

export function ContextSelector({ onContextChange, currentTimeSlot, currentIsWeekend }: ContextSelectorProps) {
  const [selectedTimeSlot, setSelectedTimeSlot] = useState(currentTimeSlot || 'morning')
  const [selectedIsWeekend, setSelectedIsWeekend] = useState(currentIsWeekend ?? false)

  const timeSlots = [
    { value: 'morning', label: 'Morning (6AM-12PM)', icon: Sun, color: 'text-yellow-500' },
    { value: 'afternoon', label: 'Afternoon (12PM-6PM)', icon: Sunset, color: 'text-orange-500' },
    { value: 'evening', label: 'Evening (6PM-10PM)', icon: Moon, color: 'text-blue-500' },
    { value: 'late_night', label: 'Late Night (10PM-6AM)', icon: CloudMoon, color: 'text-indigo-500' },
  ]

  const handleTimeSlotChange = (timeSlot: string) => {
    setSelectedTimeSlot(timeSlot)
    onContextChange(timeSlot, selectedIsWeekend)
  }

  const handleDayTypeChange = (isWeekend: boolean) => {
    setSelectedIsWeekend(isWeekend)
    onContextChange(selectedTimeSlot, isWeekend)
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5" />
          Context-Aware Recommendations
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Select your shopping context to get personalized recommendations based on CoFARS-Sparse model
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Time Slot Selection */}
        <div>
          <label className="text-sm font-medium mb-2 block">Time of Day</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {timeSlots.map((slot) => {
              const Icon = slot.icon
              const isSelected = selectedTimeSlot === slot.value
              return (
                <Button
                  key={slot.value}
                  variant={isSelected ? 'default' : 'outline'}
                  className="flex flex-col items-center gap-2 h-auto py-3"
                  onClick={() => handleTimeSlotChange(slot.value)}
                >
                  <Icon className={`h-5 w-5 ${isSelected ? '' : slot.color}`} />
                  <span className="text-xs text-center">{slot.label}</span>
                </Button>
              )
            })}
          </div>
        </div>

        {/* Day Type Selection */}
        <div>
          <label className="text-sm font-medium mb-2 block">Day Type</label>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant={!selectedIsWeekend ? 'default' : 'outline'}
              className="flex items-center gap-2"
              onClick={() => handleDayTypeChange(false)}
            >
              <Calendar className="h-4 w-4" />
              Weekday
            </Button>
            <Button
              variant={selectedIsWeekend ? 'default' : 'outline'}
              className="flex items-center gap-2"
              onClick={() => handleDayTypeChange(true)}
            >
              <Calendar className="h-4 w-4" />
              Weekend
            </Button>
          </div>
        </div>

        {/* Context Info */}
        <div className="bg-muted/50 p-3 rounded-lg">
          <p className="text-sm">
            <strong>Current Context:</strong> {selectedTimeSlot.replace('_', ' ')} on {selectedIsWeekend ? 'weekend' : 'weekday'}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            The model will recommend products based on similar users' behavior in this context
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
