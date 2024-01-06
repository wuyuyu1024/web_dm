class MapHolder {
    constructor(data, Pinv, clf, main_svg, real_svg, fake_svg) {
        this.data = data;
        // console.log(data)
        this.Pinv = Pinv;
        this.clf = clf;
        this.main_svg = main_svg;
        this.real_svg = real_svg;
        this.fake_svg = fake_svg;
        
        this.padding = data.padding

        // padding_x = (d3.max(data.X2d, d => d[0]) - d3.min(data.X2d, d => d[0])) * padding
        // padding_y = (d3.max(data.X2d, d => d[1]) - d3.min(data.X2d, d => d[1])) * padding
        // console.log(padding_x)
        // console.log(padding_y)
      
        this.scale_x = d3.scaleLinear()
          .domain([0-this.padding, 1+this.padding])
          .range([0, map_width])
      
        this.scale_y = d3.scaleLinear()
          .domain([0-this.padding, 1+this.padding])
          .range([0, map_height])
      
        this.labels_scalar = d3.scaleOrdinal()
          .domain(d3.extent(data.label, d => d))
          .range(d3.schemeCategory10)

        
    }

    init_plots(){
        this.plot_scatters()
        this.plot_Pinv_on_click()
        this.update_main_map()
        console.time('init zfinder')
        this.zfinder = new KNNRegressor(4);
        this.zfinder.fit(this.data.XY, this.data.z_of_XY)
        console.timeEnd('init zfinder')

    }

    map_showing_event(event){
        // console.log(event)
        // get value 
        const value = event.target.value
        console.log(value)
    }

    update_main_map(){
        //TODO: check some conditions
        this.plot_DMdata()
    }

    async plot_DMdata(proba=true){
        // count time
        console.time('Pinv')
        const Iinv = this.Pinv.predict([tf.tensor2d(this.data.XY, [this.data.XY.length, 2]), tf.tensor2d(this.data.z_of_XY, [this.data.z_of_XY.length, 16])])
        // time end
        console.timeEnd('Pinv')
        // time start
        console.time('clf')
        const predictions = this.clf.predict(Iinv)
        // time end
        console.timeEnd('clf')
        // time start
        console.time('process')
        const labels = await predictions.argMax(1).array();
        // console.log(labels)
        const colors = labels.map(d => this.labels_scalar(d))
        // console.log(colors)
        // time end
        console.timeEnd('process')
        // time start
        console.time('plot pixels')
        if (proba) {
            const confidences = await predictions.array().then(data => data.map(d => d3.max(d)));
            update_image(this.main_svg, colors, 4, confidences);
            // return [labels, confidences]
        }
        else {
            update_image(this.main_svg, colors, 3)}
        // time end
        console.timeEnd('plot pixels')
        }
       

    /// plot the scatters
    plot_scatters(){
        let vis = this;
        this.scatters = vis.main_svg.selectAll("circle")
            .data(vis.data.X2d)
            .enter()
            .append("circle")
            .attr("cx", d => vis.scale_x(d[0]))
            .attr("cy", d => vis.scale_y(d[1]))
            .attr("r", 4)
            .attr("fill", (d, i) => vis.labels_scalar(data.label[i]))
            .attr("stroke", "black")
            .attr("stroke-width", 0.5)
            .on("mouseover", function(d){
            d3.select(this)
                .attr("stroke-width", 1.5)
                .attr("r", 1.5*d3.select(this).attr("r"))
            })
            .on("mouseout", function(d){
            d3.select(this)
                .attr("stroke-width", 0.5)
                .attr("r", 4)
            })
            .on("click", function(evnet, x2d){
                vis.main_svg.selectAll(".selected_circle").remove()
                // console.log(event)
                // console.log(x2d)
                const index = vis.data.X2d.indexOf(x2d)
                console.log(index)
                const image = vis.data.X[index].map(d => d * 255)
                update_image(vis.real_svg, image)

                vis.main_svg.append("circle")
                        .attr("cx", vis.scale_x(x2d[0]))
                        .attr("cy", vis.scale_y(x2d[1]))
                        .attr("r", d3.select(this).attr("r"))
                        .attr("fill", d3.select(this).attr("fill"))
                        .attr("stroke", "red")
                        .attr("stroke-width", 2)
                        .attr('class', 'selected_circle')
                        .attr("pointer-events", "none")
    })
    }

    gaussianFilter(X2d, sigma = 0.1) {
        const x = this.XY_grid.slice([0, 0], [-1, 1]).reshape([-1]);
        const y = this.XY_grid.slice([0, 1], [-1, 1]).reshape([-1]);
        const x0 = X2d[0];
        const y0 = X2d[1];

        const gaussian = tf.exp(tf.neg(tf.add(tf.square(tf.sub(x, x0)), tf.square(tf.sub(y, y0))).div(2 * sigma * sigma)));
        let filter = gaussian.reshape([this.GRID, this.GRID, -1]);
        filter = tf.tile(filter, [1, 1, this.current_z.shape[1]]);
        return filter;
    }

    updateZ() {
        // Assuming encode and get_z are methods that you have defined
        const deltaZ = this.encode(this.X[this.dataInd]).sub(this.getZ(this.X2d[this.dataInd]));
        const filter = this.gaussianFilter(this.X2d[this.dataInd], this.shapeRadius);

        const delta = deltaZ.mul(filter);
        this.current_z = this.current_z.add(delta.reshape([-1, this.current_z.shape[1]]).mul(this.shapeFactor));
        this.current_z = tf.clipByValue(this.current_z, -1, 1);        }


    // get z for a given x2d
    get_z(x2d){
        // console.log('init zfinder')
        const result = this.zfinder.predict(x2d)
        // console.log(result)
        return result
        }

        // update the fake image on click (press) event
    plot_Pinv_on_click(){
        let vis = this;
        this.main_svg
            // .on('mousedown', function(event){this.mousedown = true})
            // .on('mouseup', function(event){this.mousedown = false})
            // .on("mousemove", async function(event) 
            //     {
            //     if (!this.mousedown) return;
            //     // main_svg.selectAll(".selected_circle").remove()
            //     // Get the SVG element's screen transformation matrix
            //     var CTM = main_svg.node().getScreenCTM();
            
            //     // Calculate the point clicked in the SVG's coordinate system
            //     var svgPoint = main_svg.node().createSVGPoint();
            //     svgPoint.x = event.clientX;
            //     svgPoint.y = event.clientY;
            //     var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
            
            //     // Now svgPointTransformed.x and svgPointTransformed.y are the correct coordinates
            //     // console.log(svgPointTransformed.x, svgPointTransformed.y);
            
            //     // Use your scales to invert the coordinates
            //     var x_scale = mapholder.scale_x.invert(svgPointTransformed.x);
            //     var y_scale = mapholder.scale_y.invert(svgPointTransformed.y);
            //     ////////////////
            //     const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
            //     const z_tensor = await mapholder.zfinder.predict(x2d_tensor)
            //     // // z_tensor.print()
            //     // const predict = await mapholder.Pinv.predict([x2d_tensor, z_tensor]).array()
            //     const predict = mapholder.Pinv.predict([x2d_tensor, z_tensor]).arraySync()  // not ideal
            //     // pred = clf.predict(data).arraySync()
            //     const image = predict[0].map(d => d * 255)
            //     update_image(vis.fake_svg, image)
            
            //     // console.log(z) 
            // })
            .on('click', async function(event){
                // console.log('clicking on the map')
                // main_svg.selectAll(".selected_circle").remove()
                // Get the SVG element's screen transformation matrix
                var CTM = main_svg.node().getScreenCTM();
            
                // Calculate the point clicked in the SVG's coordinate system
                var svgPoint = main_svg.node().createSVGPoint();
                svgPoint.x = event.clientX;
                svgPoint.y = event.clientY;
                var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
            
                // Now svgPointTransformed.x and svgPointTransformed.y are the correct coordinates
                // console.log(svgPointTransformed.x, svgPointTransformed.y);
            
                // Use your scales to invert the coordinates
                var x_scale = mapholder.scale_x.invert(svgPointTransformed.x);
                var y_scale = mapholder.scale_y.invert(svgPointTransformed.y);
                ////////////////
                const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
                const z_tensor = await mapholder.zfinder.predict(x2d_tensor)
                // z_tensor.print()
                const predict = await mapholder.Pinv.predict([x2d_tensor, z_tensor]).array()
                // pred = clf.predict(data).arraySync()
                const image = predict[0].map(d => d * 255)
                update_image(vis.fake_svg, image)
        })

    

}

}
