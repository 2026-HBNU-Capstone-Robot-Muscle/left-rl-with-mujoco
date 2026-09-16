"""Bounded one-direction external integral correction; motor I must be zero."""
from core import Fault, states_by_id, finite
import math

class PositionCompensation:
    def __init__(self,cfg,extra):
        if type(extra) is not int or not 0<=extra<=200:raise Fault('Compensation must be 0..200 tick')
        self.extra=extra
        self.minimum={m['id']:max(0,m['provisional_close_tick']-extra) for m in cfg['motors']}
        self.offset={i:0. for i in self.minimum}
        self.previous_goal={}
    def command(self,goals,rows,now,dt):
        by=states_by_id(rows,now,.1)
        if set(goals)!=set(self.minimum):raise Fault('Incomplete compensation goals')
        if not 0<finite(dt)<=.2:raise Fault('Invalid compensation interval')
        result={}
        for i,goal in goals.items():
            goal=finite(goal)
            error=by[i]['position_tick']-goal  # positive: needs more winding
            if goal>self.previous_goal.get(i,goal)+1:
                self.offset[i]=0.  # Do not fight a new opening policy target.
            elif error>10:
                self.offset[i]+=min(.5*error,40.)*dt
            elif error < -5:
                self.offset[i]-=40.*dt
            # Inside the deadband retain the effort needed to oppose the spring.
            self.offset[i]=max(0.,min(self.extra,goal-self.minimum[i],self.offset[i]))
            result[i]=max(self.minimum[i],math.ceil(goal-self.offset[i]))
            self.previous_goal[i]=goal
        return result
