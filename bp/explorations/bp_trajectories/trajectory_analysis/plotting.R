
plotMetric = function(values, numGroups, name='Metric') {
    assert_that(length(values) == length(numGroups))
    
    data.frame(NumGroups=numGroups, Metric=values) %>%
        ggplot(aes(NumGroups, Metric)) +
        geom_line(size=.5) +
        geom_point(size=1) +
        scale_x_continuous(breaks=seq(1, max(numGroups)), minor_breaks=NULL) +
        scale_y_continuous(breaks=pretty_breaks(5), labels=comma) +
        labs(x='Number of clusters', y=name)
}


example_plotGroupTrajectories = function(data, props) {
    if(has_attr(data, 'groupTrajs')) {
        groupTrajs = attr(data, 'groupTrajs') %>% copy()
        if(missing(props)) {
            props = data[, .(N=uniqueN(Id)), keyby=Group]$N / uniqueN(data$Id)
        }
    } else {
        groupTrajs = copy(data)
    }
    
    if(!missing(props)) {
        groupTrajs[, GroupLabel := factor(Group, levels=levels(Group), labels=sprintf('%s (%s%%)', gsub(' ', '\n', levels(Group)), round(props * 100)))]
    } else {
        groupTrajs[, GroupLabel := Group]
    }
    
    # Point positions
    times = sort(unique(groupTrajs$Time))
    nPoints = min(length(times), 10)
    pointIdx = seq(1, length(times), length.out=nPoints) %>% round
    pointData = groupTrajs[Time %in% times[pointIdx]]
    
    # Compute label positions
    library(lpSolve)
    groupTrajs[, Id := Group]
    groupMat = transformToRepeatedMeasures(groupTrajs)
    k = nlevels(groupTrajs$Group)
    labelIdx = seq(1, length(times), length.out=k+2)[2:(k+1)] %>% round
    m = apply(groupMat[, labelIdx], 2, function(x) vapply(1:k, function(i) min(abs(x[i] - x[-i])), FUN.VALUE=0))
    lp = lp.assign(m, direction='max')
    groupOrder = apply(lp$solution, 2, which.max)
    
    textData = groupTrajs[data.frame(Group=levels(groupTrajs$Group)[groupOrder],
                                     Time=times[labelIdx]), on=c('Group', 'Time')]
    
    # Plot
    ggplot(groupTrajs, aes(x=Time, y=Usage, group=Group)) +
        geom_line(size=.1) +
        geom_point(data=pointData, aes(x=Time, y=Usage, shape=GroupLabel), size=2) +
        geom_label(data=textData, aes(x=Time, y=Usage, label=Group),
                   color='gray20',
                   alpha=.9,
                   size=ifelse(nchar(levels(groupTrajs$Group)) == 1, 2.5, 1.5)) +
        scale_x_continuous(breaks=seq(1, 361, by=60)) +
        scale_y_continuous(breaks=seq(0, 12, by=2)) +
        scale_shape_manual(name='Cluster', values=1:k) +
        expand_limits(x=range(data$Time), y=c(0, 7)) +
        labs(x='Day', y='Hours of use')
}


plotGroupTrajectories = function(data, props) {
    if(has_attr(data, 'groupTrajs')) {
        groupTrajs = attr(data, 'groupTrajs') %>% copy()
        if(missing(props)) {
            props = data[, .(N=uniqueN(Id)), keyby=Group]$N / uniqueN(data$Id)
        }
    } else {
        groupTrajs = copy(data)
    }
    
    if(!missing(props)) {
        groupTrajs[, GroupLabel := factor(Group, levels=levels(Group), labels=sprintf('%s (%s%%)', gsub(' ', '\n', levels(Group)), round(props * 100)))]
    } else {
        groupTrajs[, GroupLabel := Group]
    }
    
    # Point positions
    times = sort(unique(groupTrajs$Time))
    nPoints = min(length(times), 10)
    pointIdx = seq(1, length(times), length.out=nPoints) %>% round
    pointData = groupTrajs[Time %in% times[pointIdx]]
    
    # Compute label positions
    library(lpSolve)
    groupTrajs[, Id := Group]
    groupMat = transformToRepeatedMeasures(groupTrajs)
    k = nlevels(groupTrajs$Group)
    labelIdx = seq(1, length(times), length.out=k+2)[2:(k+1)] %>% round
    m = apply(groupMat[, labelIdx], 2, function(x) vapply(1:k, function(i) min(abs(x[i] - x[-i])), FUN.VALUE=0))
    lp = lp.assign(m, direction='max')
    groupOrder = apply(lp$solution, 2, which.max)
    
    textData = groupTrajs[data.frame(Group=levels(groupTrajs$Group)[groupOrder],
                                     Time=times[labelIdx]), on=c('Group', 'Time')]
    
    # Plot
    ggplot(groupTrajs, aes(x=Time, y=mitteldruck, group=Group)) +
        geom_line(size=.1) +
        geom_point(data=pointData, aes(x=Time, y=mitteldruck, shape=GroupLabel), size=2) +
        geom_label(data=textData, aes(x=Time, y=mitteldruck, label=Group),
                    color='gray20',
                    alpha=.9,
                    size=ifelse(nchar(levels(groupTrajs$Group)) == 1, 2.5, 1.5)) +
        # scale_x_continuous(breaks=seq(1, 361, by=60)) +
        scale_y_continuous(breaks=seq(0, 250, by=2)) +
        scale_shape_manual(name='Cluster', values=1:k) +
        expand_limits(x=range(data$Time), y=c(0, 7)) +
        labs(x='Day', y='BP (mmHg)')
}
