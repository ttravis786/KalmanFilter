# find all possible seeds for a point

def next_point(df, point, max_dist=3.0, max_time=0.3):
    x, y, t = point
    df_suitable = df[(df.time <= t + max_time) &
                     (df.time <= t + max_time) &
                     (((((df.x - x)**2 +
                         (df.y-y))**2)**0.5) <= max_dist)]
    # try all closest in time and distance 2nd order
    df_suitable = df_suitable
    # if fail try next lot
    # assume time assecding
    return ()

