class MapHolder {
    constructor(data, Pinv, clf, main_svg, real_svg, fake_svg) {
        this.data = data;
        console.log(data)
        this.Pinv = Pinv;
        this.clf = clf;
        this.main_svg = main_svg;
        this.real_svg = real_svg;
        this.fake_svg = fake_svg;
        this.XY_tensor = tf.tensor2d(data.XY, [data.XY.length, 2]);
        this.current_z = tf.tensor2d(data.z_of_XY, [data.z_of_XY.length, data.z_of_XY[0].length]);
        this.initial_z = this.current_z.clone();
        
        this.padding = data.padding
        this.adjusting = false
        this.adjust_factor = 0.3
        this.map_showing = 0
        this.hold_checkbox_event = this.hold_checkbox_event.bind(this); // WTF is this?
        // this.make_moving_circle = this.make_moving_circle.bind(this);
        this.map_showing_event = this.map_showing_event.bind(this);
        this.updateZ = this.updateZ.bind(this);


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
        
          this.zfinder = new KNNRegressor(4);
          this.zfinder.fit(this.XY_tensor, this.current_z) // TODO: use tensor directly
    }

    init_plots(){
        this.plot_scatters()
        this.plot_Pinv_on_click()
        this.update_main_map()
        console.time('init zfinder')
        
        console.timeEnd('init zfinder')
        this.make_moving_circle()

    }



    update_main_map(){
        //TODO: check some conitions
        if (this.map_showing == 0) {
            this.plot_DMdata(false)
        }
        else if (this.map_showing == 1) {
            this.plot_DMdata(true)
        }
        else if (this.map_showing == 2) {
            console.log('placeholder')
            this.plot_diff_map()
        }
        else if (this.map_showing == 3) {
            
            const tem = [['FFFFFF']]
            update_image(this.main_svg, tem, 4, [0])
        }
  
    }

    async plot_DMdata(proba=true){
        // count time
        console.time('Pinv')
        const Iinv = this.Pinv.predict([this.XY_tensor, this.current_z])
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
        const label_colors = labels.map(d => this.labels_scalar(d))
        // console.log(label_colors)
        // time end
        console.timeEnd('process')
        // time start
        console.time('plot pixels')
        if (proba) {
            // Find the maximum values along the same axis
            const maxValuesTensor = predictions.max(1);
            // Convert the tensor to array if you need to work with regular JavaScript arrays
            const confidences = await maxValuesTensor.array();
            update_image(this.main_svg, label_colors, 4, confidences);
            // return [labels, confidences]
        }
        else {
            update_image(this.main_svg, label_colors, 3)}
        // time end
        console.timeEnd('plot pixels')
        }
       
    async plot_diff_map(){
        let diff = tf.sub(this.current_z, this.initial_z)
        // l2 norm
        diff = diff.pow(2).sum(1).sqrt()
        let diff_cpu = await diff.array()
        // console.log(diff_cpu)
        let diff_scale = d3.scaleLinear()
            .domain(d3.extent(diff_cpu))
            .range([0, 255])
        // console.log(diff_scale(0.5))
        diff_scale = diff_cpu.map(d => diff_scale(d))
        update_image(this.main_svg, diff_scale, 1)

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
            .on("click", async function(evnet, x2d){
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
                // update z
                if (vis.adjusting) {
                    await vis.updateZ(index)
                    vis.update_main_map()
                }
            
    })
    }

    gaussianFilter(X2d, sigma = 0.2) {
        console.log('gaussianFilter')
        console.log(sigma)
        const x0 = X2d[0];
        const y0 = X2d[1];
        const x = this.XY_tensor.slice([0, 0], [-1, 1]).reshape([-1]);
        const y = this.XY_tensor.slice([0, 1], [-1, 1]).reshape([-1]);
        x.print()
    
        const gaussian = tf.exp(tf.neg(tf.add(tf.square(tf.sub(x, x0)), tf.square(tf.sub(y, y0))).div(2 * sigma * sigma)));
        // gaussian.print()
        // console.log(gaussian.shape)
        

        let filter = gaussian.reshape([this.data.GRID, this.data.GRID, -1]);

        // let gaussian_image = filter.reshape([10000,]).arraySync()
        // gaussian_image = gaussian_image.map(d => d * 255)
        // update_image(this.main_svg, gaussian_image, 1) // looks correct

        // filter.print()
        filter = tf.tile(filter, [1, 1, this.data.z[0].length]); // (100,100, 16)
        console.log(filter.shape)
        return filter;
    }

    async updateZ(index) {
        let vis = this
        console.log('update z')
        // console.log(index)
        // console.log(vis)
        // Assuming encode and get_z are methods that you have defined
        let X2d = vis.data.X2d[index]
        console.log('check 2D location')
        console.log(X2d)
        let x2d_tensor = tf.tensor2d(X2d, [1, 2])
        let z_tensor = await vis.zfinder.predict(x2d_tensor)

        let target_z = vis.data.z[index]
        const deltaZ = tf.sub(target_z, z_tensor);
        deltaZ.print()
        const filter = vis.gaussianFilter(X2d, this.radius); // not working

        const delta = deltaZ.mul(filter);
        console.log(delta.shape)
        // console.log(this.adjust_factor)
        this.current_z = this.current_z.add(delta.reshape([-1, this.current_z.shape[1]]).mul(this.adjust_factor));
        // this.current_z = this.current_z.add(delta.reshape([10000,1, 1]).mul(this.adjust_factor));
        // this.current_z = tf.clipByValue(this.current_z, -1, 1);  //// WTF is this? it ruins everything
        this.zfinder.fit(this.XY_tensor, this.current_z)
    }


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
                // // Get the SVG element's screen transformation matrix
                // var CTM = main_svg.node().getScreenCTM(); // already in main
            
                // // Calculate the point clicked in the SVG's coordinate system
                // var svgPoint = main_svg.node().createSVGPoint(); // already in main
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
 
        // events handling
        map_showing_event(event){
            console.log(this)
            // get value 
            const value = event.target.value
            console.log(value)
            this.map_showing = value
            console.log(this.map_showing)
            this.update_main_map()

        }
    
        hold_checkbox_event(event){
            console.log(event)
            // get value 
            const value = event.target.checked
            console.log(value)
            this.adjusting = value
        }
    
        radius_slider_event(event){
            console.log(event)
            // get value 
            const value = event.target.value
            console.log(value)
            this.radius = value
        }
    
        factor_slider_event(event){
            console.log(event)
            // get value 
            const value = event.target.value
            console.log(value)
            this.factor = value // to fix
        }

        make_moving_circle = () => {
            let vis = this
            // console.log(vis)
            // let radius_to_pixel_x = vis.scale_x(radius) - vis.scale_x(0)  //////SCALER CAN NOT BE USED HERE
            // console.log('debug')
            // console.log(radius_to_pixel_x)
            // let radius_to_pixel_y = this.scale_y(radius) - this.scale_y(0)
            let radius_to_pixel_x = 40
            this.moving_circle = vis.main_svg.append("circle")
            // console.log(this.moving_circle)

            this.moving_circle
                .attr("r", radius_to_pixel_x) // todo this is a circle now
                .attr("fill", "#00000000")
                .attr("stroke", 'white')
                .attr("stroke-width", 2)
                .attr("class", "moving_circle")
                // no emit event for this circle
                .attr("pointer-events", "none")
                // style for this circle: dashed
                .style("stroke-dasharray", ("3, 5"))


            vis.main_svg
            .on("mouseout",  (event) => {  // Arrow function here
                if (vis.adjusting == false) {this.moving_circle
                                                    .attr("cx", -30)
                                                    .attr("cy", -30);}
                else {this.moving_circle                    
                    // .transition()
                    // .duration(300)
                    .attr("cx", 0.5*map_width)
                    .attr("cy", 0.5*map_height)

                }
                
            })  
            .on("mousemove", (event) => { 
                    // read the checkbox
                // console.log(vis.adjusting)
                if (vis.adjusting == false) {return}
                svgPoint.x = event.clientX;
                svgPoint.y = event.clientY;
                var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
                this.moving_circle//.raise()
                    .attr("cx", svgPointTransformed.x)
                    .attr("cy", svgPointTransformed.y)
            })
                }
}
