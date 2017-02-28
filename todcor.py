# Implementation of TODOC algorithm.
# Adapted from https://github.com/ajriddle/Work-Files/blob/master/cps_fcns.py.

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, leastsq


def create_cross_correlations(observation, template1, template2):
    """ Create cross corelations c1, c2, c12 to be used for todcor.

    Inputs:
        The 3 spectrum objects.
        observation: Observation or combined spectra
        template1,
        template2

    Outputs:
        Cross correlations c1, c2, c12
    For now only doing one chip at a time, one order.
    """

    # The dictionary keys are
    spectra = {"science": observation, "template1": template1, "template2": template2}
    # can be accessed like spectra["science"].flux


    # Not yet sure what these variables are for
    r = 1.28 / 2.998E5  # Use velocity spacing of 1.28 km/s/pixel
    s = np.zeros(1)
    t1 = np.zeros(4021)
    t2 = np.zeros(4021)

    # Normalize to 0 continuum, rather than 1
    spectra['science'].flux -= 1
    spectra['template1'].flux -= 1
    spectra['template2'].flux -= 1

    # Interpolate to common logarithmic wavelength scale
    # I am very uncertian about this implementation !!!!!!!!!
    fs = interp1d(spectra['science'].xaxis, spectra['science'].flux, \
            fill_value=0., bounds_error=False)
    ft1 = interp1d(spectra['template1'].xaxis, spectra['template1'].flux, \
            fill_value=0., bounds_error=False)
    ft2 = interp1d(spectra['template2'].xaxis, spectra['template2'].flux, \
            fill_value=0., bounds_error=False)
    # I am very uncertian about this implementation !!!!!!!!!
    wv = spectra['science'].xaxis[0]*(1.+r)**np.linspace(0., 4020., 4021)
    if len(s)>1:
            s = vstack((s, fs(wv)))
            t1 = vstack((t1, ft1(wv)))
            t2 = vstack((t2, ft2(wv)))
    elif len(s)==1:
            s = fs(wv)
            t1 = ft1(wv)
            t2 = ft2(wv)

    print("lenght of s", len(s))
    # Calculate normalized 1-D cross-correlation functions for each order
    # c1 = np.zeros([len(s),8041])
    # c2 = np.zeros([len(s),8041])
    # c12 = np.zeros([len(s),8041])
    c1 = np.zeros([1, 8041])
    c2 = np.zeros([1, 8041])
    c12 = np.zeros([1, 8041])
    shift = range(-4020, 4021)
    # for i in range(len(s)):
    # for i in range(1):
        # CCF of science with primary template
    #    c1[i]=np.correlate(s[i],t1[i],mode='full')/ \
    #    ( np.sqrt(np.mean(s[i]**2))* np.sqrt(np.mean(t1[i]**2))*len(s[i]))
    c1 = np.correlate(s ,t1 ,mode='full')/ \
    ( np.sqrt(np.mean(s **2))* np.sqrt(np.mean(t1 **2))*len(s ))

        # CCF of science with secondary template
        # c2[i]=np.correlate(s[i],t2[i],mode='full')/ \
        # ( np.sqrt(np.mean(s[i]**2))* np.sqrt(np.mean(t2[i]**2))*len(s[i]))
    c2 =np.correlate(s ,t2 ,mode='full')/ \
    ( np.sqrt(np.mean(s **2))* np.sqrt(np.mean(t2 **2))*len(s ))

        # CCF between templates
        # c12[i]=np.correlate(t1[i],t2[i],mode='full')/ \
        #( np.sqrt(np.mean(t1[i]**2))* np.sqrt(np.mean(t2[i]**2))*len(t1[i]))
    c12 =np.correlate(t1 ,t2 ,mode='full')/ \
    ( np.sqrt(np.mean(t1 **2))* np.sqrt(np.mean(t2 **2))*len(t1 ))


    return c1, c2, c12, spectra




def todcor(c1, c2, c12, pshift, sshift, images):
    def gauss(x, *p):
        A, mu, sigma = p
        return A * exp(-(x-mu)**2/(2.*sigma**2))
    def gauss2D(height_x, center_x, width_x, height_y, center_y, width_y):
        return lambda x, y: height_x*exp(-((center_x-x)/width_x)**2/2)+ \
           height_y*exp(-((center_y-y)/width_y)**2/2)
    # Calculate velocity shift array using 1.28 km/s/pixel
    v = range(-4020, 4021)
    v = [1.28 * vel for vel in v]

    # Calculate telluric offset using order near 7500 A
    # delta1 = list(c1[23]).index(max(c1[23]))
    # delta2 = list(c2[23]).index(max(c2[23]))
    # delta1 = v[delta1]
    # delta2 = v[delta2]

    # Account for differences in heliocentric velocities
    # delta1-=images['science']['helio']-images['temp1']['helio']
    # delta2-=images['science']['helio']-images['temp2']['helio']

    delta1 = 1  # Fix delta jsut to put something as a test
    delta2 = 1
    print('cps deltas =',delta1, delta2)

    # Remove orders that provide unreliabe RVs
    # orders = list(images['science']['w'][:,0])
    orders=[images['science'].xaxis[:]]
    print("order length", len(orders))

    # for wave in orders:
    #    if abs(wave-5801)<15 or abs(wave-5898)<15 or abs(wave-6553)<15 or \
    #    abs(wave-6797)<15 or abs(wave-7213)<15 or abs(wave-7520)<15:
            # i = orders.index(wave)
            # c1 = vstack(([c1[:i],c1[i+1:]]))  # removing order i
            # c2 = vstack(([c2[:i],c2[i+1:]]))
            # c12 = vstack(([c12[:i],c12[i+1:]]))
            # orders.pop(i)

    print('Calculating TODCOR...')
    # Calculate TODCOR
    R = np.zeros(len(orders),dtype=list)
    m = np.zeros([len(orders),3])
    sp = int(pshift)
    ss = int(sshift)
    # Set size of todcor box in velocity space (needs to be an odd number).
    # Processing time scales as size^2. E.g. vrange=79 will search in a 50 km/s
    # by 50 km/s box centered on the inital guess values for the primary and
    # secondary shifts.
    vrange = 79  # 157
    # for k in range(len(R)):
    #     R[k]=np.zeros([vrange, vrange])
    #     for i in range(vrange):
    #         s2 = i+4020+ss-(vrange-1)/2
    #         for j in range(vrange):
    #             s1 = j+4020+sp-(vrange-1)/2
    #             R[k][i, j]= np.sqrt((c1[k, s1]**2-2*c1[k, s1]*c2[k, s2]* \
    #                 c12[k, s2-s1+4020]+c2[k, s2]**2)/(1-c12[k, s2-s1+4020]**2))
    #             if R[k][i, j]>m[k, 2]:
    #                 m[k]=[i, j, R[k][i, j]]
    #     R[k][np.isnan(R[k])]=0.
    #     R[k][np.isinf(R[k])]=0.

    R = np.zeros([vrange, vrange])
    for i in range(vrange):
        s2 = i+4020+ss-(vrange-1)/2
        for j in range(vrange):
            s1 = j+4020+sp-(vrange-1)/2
            R[i, j] = np.sqrt((c1[s1]**2-2*c1[s1]*c2[s2] * \
                c12[s2-s1+4020]+c2[s2]**2)/(1-c12[s2-s1+4020]**2))
            # if R[i, j]>m[2]:
            #    m=[i, j, R[i, j]]
    R[np.isnan(R)] = 0.
    R[np.isinf(R)] = 0.

    print('Calculating velocity errors...')
    # Convert to velocity units
    v1 = images['template1'].rv
    v2 = images['template2'].rv
    vp = list(range(int(-(vrange-1)/2),int((vrange-1)/2+1)))
    vs = list(range(int(-(vrange-1)/2),int((vrange-1)/2+1)))
    for i in range(len(vp)):
        vp[i] = round((vp[i]+sp)*1.28, 2)+v1
        vs[i] = round((vs[i]+ss)*1.28, 2)+v2

    fwhm = {'p': [],'s': []}
    peak = {'p': [],'s': []}
    error = {'p': [],'s': []}
    center = {'p': [],'s': []}
    # for i in range(len(m)):
    p1=[.05, v1+(sp-(vrange-1)/2+m)*1.28, 10]
    p2=[.05, v2+(ss-(vrange-1)/2+m)*1.28, 10]
    try:
            coeff1, var_matrix1 = curve_fit(gauss, vp, R[m[0], :]- \
                min(R[m[0], :]), p0=p1)
            coeff2, var_matrix2 = curve_fit(gauss, vs, R[:, m[1]]- \
                min(R[:, m[1]]), p0=p2)
    except RuntimeError:
            fwhm['p'].append('')
            fwhm['s'].append('')
            peak['p'].append('')
            peak['s'].append('')
            error['p'].append('')
            error['s'].append('')
            center['p'].append('')
            center['s'].append('')
# #            X, Y = meshgrid(vp, vs)
# #            hf = figure()
# #            ha = hf.add_subplot(111, projection='3d')
# #            ha.plot_surface(X, Y, R[i],cmap=cm.cubehelix)
# #            xlabel('Primary Velocity')
# #            ylabel('Secondary Velocity')
# #            title(r'Order starting $\lambda$ = %f A' % orders[i])
# #            show()
    else:
            fwhm['p'].append(2.355*coeff1[2])
            fwhm['s'].append(2.355*coeff2[2])
            peak['p'].append(coeff1[0])
            peak['s'].append(coeff2[0])
            center['p'].append(coeff1[1])
            center['s'].append(coeff2[1])
            fit1 = gauss(vp,*coeff1)
            fit2 = gauss(vs,*coeff2)
            # plot(vp, R[i][m[i, 0],:]-min(R[i][m[i, 0],:]),'b')
            # plot(vp, fit1,'--r',label='fit')
            # xlabel('Primary Velocity (km/s)')
            # title(r'Order starting $\lambda$ = %f A' % orders[i])
            # ylabel('Correlation')
            # legend(loc='best')
            # print m[8]
            # savefig('ccf_fit.pdf')
            # show()
            # plot(vs, R[i][:,m[i, 1]]-min(R[i][:,m[i, 1]]),'b')
            # plot(vs, fit2,'--r',label='fit')
            # xlabel('Secondary Velocity (km/s)')
            # ylabel('Correlation')
            # show()
            # close()
            out1 = list(R[i][m[i, 0],:m[i, 1]-17])+list(R[i][m[i, 0],m[i, 1]+18:])
            out2 = list(R[i][:m[i, 0]-17, m[i, 1]])+list(R[i][m[i, 0]+18:,m[i, 1]])
            std1 = std(out1)
            std2 = std(out2)
            r1 = peak['p'][-1] / (np.sqrt(2)*std1)
            r2 = peak['s'][-1] / (np.sqrt(2)*std2)
            sig1 = (3./8.)*fwhm['p'][-1]/(1+r1)
            sig2 = (3./8.)*fwhm['s'][-1]/(1+r2)
            error['p'].append(sig1)
            error['s'].append(sig2)
            fits = {'fwhm': fwhm,'amp': peak,'center': center}
            p = [peak['p'][-1], center['p'][-1], \
               fwhm['p'][-1]/2.355, peak['s'][-1], \
               center['s'][-1], fwhm['s'][-1]/2.355]
# #            # Plot TODCOR
# #            X, Y = meshgrid(vp, vs)
# #            Z = np.zeros([vrange, vrange])
# #            f = gauss2D(*     p)
# #            for k in range(vrange):
# #                for j in range(vrange):
# #                    Z[k, j]=f(X[k, j],Y[k, j])
# #            hf = figure()
# #            ha = hf.add_subplot(111, projection='3d')
# #            ha.plot_surface(X, Y, R[i],cmap=cm.cubehelix)
# #            ha.plot_wireframe(X, Y, Z)
# #            xlabel('Primary Velocity')
# #            ylabel('Secondary Velocity')
# #            title(r'Order starting at $\lambda$ = %f A' % orders[i])
# #            show()
# #    params=[.4, m[1, 1],fwhm['p'][1]/2.355,.3, m[1, 0],fwhm['s'][1]/2.355]
# #    errorfunction = lambda p: ravel(gauss2D(*p)(*indices(shape(R[1])))-R[1])
# #    p, success = leastsq(errorfunction, params)

# #    p0=[.2, m[1, 1],10.,.1]
# #    coeff, cov = curve_fit(gaussc, range(-78, 79),R[1][20,:],p0=p0)
# #    print coeff
# #    print cov



# #    ylab=[]
# #    xlab=[]
# #    for i in range(8):
# #        ylab.append(v2+round((ss-(vrange-1)/2+20*i)*1.28, 1))
# #        xlab.append(v1+round((sp-(vrange-1)/2+20*i)*1.28, 1))
# #
# #
# #    fig = figure()
# #    ax1 = fig.add_subplot(111)
# #    index = orders.index(5618.333593)
# #    ax1.imshow(R[index],cmap=cm.cubehelix)
# #    im = ax1.imshow(R[index],cmap=cm.cubehelix)
# #    fig.colorbar(im)
# #    ax1.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
# #    ax1.set_yticklabels(ylab)
# #    ax1.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
# #    ax1.set_xticklabels(xlab)
# #    xlabel(r'$v_1$ (km/s)')
# #    ylabel(r'$v_2$ (km/s)')
# #    scatter(m[index, 1],m[index, 0],c='k',marker='+')
# #    title('Colormap Plot of R')
# #    savefig('TODCOR_Plot2.pdf')
# #    close()

    # Calculate centroid of R
    print('Calculating centroid...')
    centroids = np.zeros([len(R), 4])
    for i in range(len(R)):
        columns = np.zeros(len(R[i]))
        rows = np.zeros(len(R[i]))
        for j in range(len(R[i])):
            rows[j] = sum(R[i][j])
            columns[j] = sum(R[i][:, j])
            centroids[i, 0] += columns[j] * j
            centroids[i, 1] += rows[j] * j
        centroids[i, 0] /= sum(columns)
        centroids[i, 1] /= sum(rows)
        centroids[i, 0] = v1+(sp-(vrange-1)/2+centroids[i, 0])*1.28-delta1
        centroids[i, 1] = v2+(ss-(vrange-1)/2+centroids[i, 1])*1.28-delta2

# #    f = open('rv_results', 'a')
    g = open('current_rvs', 'a')
# #    print >>f, images['science']['target'], images['science']['hjd'],1, 1
    print >>g, images['science']['h']['targname'], images['science']['hjd'], 1, 1
    for i in range(len(m)):
        if error['p'][i]!='':
            centroids[i, 2]=error['p'][i]
            centroids[i, 3]=error['s'][i]
            m[i, 1]=v1+(sp-(vrange-1)/2+m[i, 1])*1.28-delta1
            m[i, 0]=v2+(ss-(vrange-1)/2+m[i, 0])*1.28-delta2
# #            print >>f, orders[i],m[i, 1],error['p'][i],m[i, 0],error['s'][i]
# #            print >>g, orders[i],center['p'][i],error['p'][i],m[i, 0], \
# #                  error['s'][i]
# #            print >>g, orders[i],center['p'][i]-delta1, error['p'][i], \
# #                  center['s'][i]-delta2, error['s'][i]

# #    print >>g, 'Centroids'
    for i in range(len(m)):
        if error['p'][i]!='':
            print >>g, orders[i], centroids[i, 0], centroids[i, 2], centroids[i, 1], \
                  centroids[i, 3]
# #    f.close()
    g.close()

    return R, m, fits, vp, vs, centroids
