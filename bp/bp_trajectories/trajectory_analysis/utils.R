
transformToAverage = function(data, binSize=14) {
    bins = seq(min(data$Time), max(data$Time), by=binSize)
    nBins = length(bins) - 1
    bindata = data[, .(Usage=mean(Usage)), keyby=.(Group, Id, Bin=findInterval(Time, bins, all.inside=TRUE))] %>%
        .[, Time := bins[Bin]]
    
    groupTrajs = attr(data, 'groupTrajs')
    bingroupTrajs = groupTrajs[, .(Usage=mean(Usage)), keyby=.(Group, Bin=findInterval(Time, bins, all.inside=TRUE))] %>%
        .[, Time := bins[Bin]]
    
    setattr(bindata, 'groupTrajs', bingroupTrajs)
    return(bindata[])
}

example_transformToRepeatedMeasures = function(data) {
    assert_that(is.data.frame(data), has_name(data, c('Id', 'Time', 'Usage')))
    dtWide = dcast(data, Id ~ Time, value.var='Usage')
    
    dataMat = as.matrix(dtWide[, -'Id'])
    assert_that(nrow(dataMat) == uniqueN(data$Id), ncol(dataMat) == uniqueN(data$Time))
    rownames(dataMat) = dtWide$Id
    colnames(dataMat) = names(dtWide)[-1]
    return(dataMat)
}

transformToRepeatedMeasures = function(data) {
    assert_that(is.data.frame(data), has_name(data, c('Id', 'Time', 'mitteldruck')))
    dtWide = dcast(data, Id ~ Time, value.var='mitteldruck')
    
    dataMat = as.matrix(dtWide[, -'Id'])
    assert_that(nrow(dataMat) == uniqueN(data$Id), ncol(dataMat) == uniqueN(data$Time))
    rownames(dataMat) = dtWide$Id
    colnames(dataMat) = names(dtWide)[-1]
    return(dataMat)
}



# Average posterior probability of assignments (APPA)
appa = function(pp) {
    rowMaxs(pp) %>% mean()
}

entropy = function(pp) {
    assert_that(is.matrix(pp), min(pp) >= 0, max(pp) <= 1)
    pp = pmax(pp, .Machine$double.xmin)
    -sum(rowSums(pp * log(pp)))
}

relativeEntropy = function(pp) {
    N = nrow(pp)
    K = ncol(pp)
    1 - entropy(pp) / (N * log(K))
}
