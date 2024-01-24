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
        this.initial_Iinv =  this.Pinv.predict([this.XY_tensor, this.initial_z])
        this.current_Iinv = this.initial_Iinv.clone()
        
        this.padding = data.padding
        this.adjusting = false
        this.adjust_factor = document.getElementById("slider_factor").value
        this.setting_ob = null
        this.ob_2d = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        this.ob_setted = [false, false, false, false, false, false]

        this.map_showing = document.getElementById("map_showing_dropdown").value
        this.hold_checkbox_event = this.hold_checkbox_event.bind(this); // WTF is this?
        // this.make_moving_circle = this.make_moving_circle.bind(this);
        this.map_showing_event = this.map_showing_event.bind(this);
        this.updateZ = this.updateZ.bind(this);
        this.radius_slider_event = this.radius_slider_event.bind(this);
        this.radius = document.getElementById("slider_radius").value
      
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
        this.init_ob_windows()

    }

    init_ob_windows(){
        let vis = this
        const ob_svgs = d3.selectAll(".ob_svg")
        ob_svgs.on("click", function(event, i){
            const ind = this.id.slice(2) 
            // console.log(ind)
            d3.select('.selected').classed('selected', false)
            vis.setting_ob = ind
            console.log(vis.setting_ob)
            // set boader color to read
            d3.select('#'+this.id).classed('selected', true)
            // attr('style', 'border: 2px solid red;')
         
            // this.attr('style', 'border: 2px solid red;')
        })
    }

    update_main_map(Inv=true){
              
        //TODO: check some conitions
        if (this.map_showing == 0) {
            this.plot_DMdata(false)
        }
        else if (this.map_showing == 1) {
            this.plot_DMdata(true)
        }
        else if (this.map_showing == 2) {
            console.log('placeholder')
            this.plot_diff_map_z()
        }
        else if (this.map_showing == 3) {
            console.log('placeholder2')
            this.plot_diff_map_Iinv()
        }
        else if (this.map_showing == 4) {
            
            this.main_svg.selectAll(".pixel").remove()
            this.main_svg.append('rect')
                .attr('class', 'pixel')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', map_width)
                .attr('height', map_height)
                .attr('fill', '#999999')
                .attr('opacity', 0.5)
                // to the bottom
                .lower()
  
    }
    }

    async plot_DMdata(proba=true){
        console.time('Pinv')
          this.current_Iinv = this.Pinv.predict([this.XY_tensor, this.current_z])  //  reduce reapating
        console.timeEnd('Pinv')

        console.time('clf')
        const predictions = this.clf.predict(this.current_Iinv)
        console.timeEnd('clf')

        console.time('process')
        const labels = await predictions.argMax(1).array();
        const label_colors = labels.map(d => this.labels_scalar(d))
        console.timeEnd('process')

        console.time('plot pixels')
        if (proba) {
            // Find the maximum values along the same axis
            const maxValuesTensor = predictions.max(1);
            // Convert the tensor to array if you need to work with regular JavaScript arrays
            const confidences = await maxValuesTensor.array();
            update_image(this.main_svg, label_colors, 4, confidences);
        }
        else {
            update_image(this.main_svg, label_colors, 3)}
        console.timeEnd('plot pixels')
        }
       
    async plot_diff_map_z(){
        let diff = tf.sub(this.current_z, this.initial_z)
        diff = diff.pow(2).sum(1).sqrt()  // l2 norm
        let diff_cpu = await diff.array()
        let diff_scale = d3.scaleLinear()
            .domain(d3.extent(diff_cpu))
            .range([0, 255])
        diff_scale = diff_cpu.map(d => diff_scale(d))
        update_image(this.main_svg, diff_scale, 1)

    }

    async plot_diff_map_Iinv(){
        this.current_Iinv = this.Pinv.predict([this.XY_tensor, this.current_z]) // TODO: reduce repeating
        let diff = tf.sub(this.current_Iinv, this.initial_Iinv)
        diff = diff.pow(2).sum(1).sqrt()  // l2 norm
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
                    vis.update_ob_windows()
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
    
        const gaussian = tf.exp(tf.neg(tf.add(tf.square(tf.sub(x, x0)), tf.square(tf.sub(y, y0))).div(2 * sigma * sigma)));
        let filter = gaussian.reshape([this.data.GRID, this.data.GRID, -1]);
        filter = tf.tile(filter, [1, 1, this.data.z[0].length]); // (100,100, 16)
        return filter;
    }

    async updateZ(index) {
        let vis = this
        console.log('update z')

        let X2d = vis.data.X2d[index]
        let x2d_tensor = tf.tensor2d(X2d, [1, 2])
        let z_tensor = await vis.zfinder.predict(x2d_tensor)

        let target_z = vis.data.z[index]
        const deltaZ = tf.sub(target_z, z_tensor);
        const filter = vis.gaussianFilter(X2d, this.radius); // not working

        const delta = deltaZ.mul(filter);
        const adjust_factor = +document.getElementById('slider_factor').value
        this.current_z = this.current_z.add(delta.reshape([-1, this.current_z.shape[1]]).mul(adjust_factor));
        // refit the zfinder
        this.zfinder.fit(this.XY_tensor, this.current_z)
    }

    async update_ob_windows() {
        
        let vis = this
        const ob_2d_tensor = tf.tensor2d(vis.ob_2d, [6, 2])
        const ob_z = await vis.zfinder.predict(ob_2d_tensor)
        // ob_z.print()
        const ob_Iinv = await vis.Pinv.predict([ob_2d_tensor, ob_z]).array()
        // console.log(ob_Iinv)
        
        for (let i = 0; i < 6; i++) {
            if (vis.ob_setted[i]) {
                const ob_image = ob_Iinv[i].map(d => d * 255)
                const ob_svg = d3.select("#ob" + (i))
                update_image(ob_svg, ob_image)
            }
        }
    }
    // // get z for a given x2d  // not using | can be deleted
    // get_z(x2d){
    //     // console.log('init zfinder')
    //     const result = this.zfinder.predict(x2d)
    //     // console.log(result)
    //     return result
    //     }

        // update the fake image on click (press) event
    plot_Pinv_on_click(){
        let vis = this;


        vis.main_svg
            .on('mousedown', function(event){vis.mousedown = true;})
            .on('mouseup', function(event){vis.mousedown = false;})
            // .on("mousemove", function(event) 
            //     {
        
                // if (!vis.mousedown) return;
                // const now = Date.now();
                // console.log('moving')
                // if (now - lastExecution > throttleDuration) {
                //     console.log('reach')
                //     lastExecution = now;
                //     //  event handling logic here
                //     svgPoint.x = event.clientX;
                //     svgPoint.y = event.clientY;
                //     var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
                    
                //     // Use your scales to invert the coordinates
                //     var x_scale = vis.scale_x.invert(svgPointTransformed.x);
                //     var y_scale = vis.scale_y.invert(svgPointTransformed.y);
                //     ////////////////
                //     const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
                //     const z_tensor = await vis.zfinder.predict(x2d_tensor)
                //     // z_tensor.print()
                //     const predict = await vis.Pinv.predict([x2d_tensor, z_tensor]).array()
                //     // pred = clf.predict(data).arraySync()
                //     const image = predict[0].map(d => d * 255)
                //     update_image(vis.fake_svg, image)
                // }
            // })
            .on('click', async function(event){
                console.log('clicked')
                svgPoint.x = event.clientX;
                svgPoint.y = event.clientY;
                var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
                
                // Use your scales to invert the coordinates
                var x_scale = vis.scale_x.invert(svgPointTransformed.x);
                var y_scale = vis.scale_y.invert(svgPointTransformed.y);
                ////////////////
                const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
                const z_tensor = await vis.zfinder.predict(x2d_tensor)
                // z_tensor.print()
                const predict = await vis.Pinv.predict([x2d_tensor, z_tensor]).array()
                // pred = clf.predict(data).arraySync()
                const image = predict[0].map(d => d * 255)
                update_image(vis.fake_svg, image)

                console.log(vis.setting_ob)
                if (vis.setting_ob !== null) {
                    d3.select(".ob_text" + vis.setting_ob).remove()
                    vis.ob_2d[vis.setting_ob] = [x_scale, y_scale]
                    vis.ob_setted[vis.setting_ob] = true
                    
                    vis.update_ob_windows()
                    // add a text marker
                    vis.main_svg.append('text')
                        .attr('x', svgPointTransformed.x)
                        .attr('y', svgPointTransformed.y)
                        .text(vis.setting_ob)
                        .attr('class', 'ob_text' + vis.setting_ob)
                        .attr('pointer-events', 'none')
                        .attr('text-anchor', 'middle')
                        .attr('fill', 'white')
                    d3.select("#ob"+vis.setting_ob).classed('selected', false)
                    vis.setting_ob = null
                }
        })   
}
 
        // events handling
        map_showing_event(event){
            const value = event.target.value
            this.map_showing = value
            // console.log(this.map_showing)
            this.update_main_map()

        }
    
        hold_checkbox_event(event){
            console.log(event)
            const value = event.target.checked
            console.log(value)
            this.adjusting = value
            this.moving_circle.attr("cx", 0.5*map_width)
            .attr("cy", 0.5*map_height)
        }
    
        radius_slider_event(event){
            console.log(event)
            // get value 
            const value = event.target.value
            console.log(value)
            this.radius = value
            this.radius_to_pixel_x = this.scale_x(value) - this.scale_x(0)  //////SCALER CAN NOT BE USED HERE
            this.moving_circle.attr("r", this.radius_to_pixel_x*2)
            // .attr("cx", 0.5*map_width)
            // .attr("cy", 0.5*map_height)
        }
    

        make_moving_circle = () => {
            let vis = this

            let lastExecution = 0;
            const throttleDuration = 80; //
            // console.log(vis)
            // let radius_to_pixel_x = vis.scale_x(radius) - vis.scale_x(0)  //////SCALER CAN NOT BE USED HERE
            // console.log('debug')
            // console.log(radius_to_pixel_x)
            // let radius_to_pixel_y = this.scale_y(radius) - this.scale_y(0)
            const slider_value = document.getElementById("slider_radius").value
            let radius_to_pixel_x = slider_value * map_width *2
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
                                                    .attr("cx", 1000)
                                                    .attr("cy", 1000)
                                                    // .attr("r", 0)
                                                }
                else {this.moving_circle                    
                    // .transition()
                    // .duration(300)
                    .attr("cx", 0.5*map_width)
                    .attr("cy", 0.5*map_height)

                }
                
            })  
            .on("mousemove", async (event) => { 
                if (vis.adjusting == true)
                {svgPoint.x = event.clientX;
                svgPoint.y = event.clientY;
                var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
                this.moving_circle//.raise()
                    .attr("cx", svgPointTransformed.x)
                    .attr("cy", svgPointTransformed.y)
                    // .attr("r", this.radius_to_pixel_x *2)
                }
                    // read the checkbox
                // console.log(vis.adjusting)
                const now = Date.now();
                // console.log('moving')
                if (vis.mousedown && Date.now() - lastExecution > throttleDuration) {
                    // console.log('reach')
                    lastExecution = now;
                    
                    svgPoint.x = event.clientX;
                    svgPoint.y = event.clientY;
                    var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
                    
                    // Use your scales to invert the coordinates
                    var x_scale = vis.scale_x.invert(svgPointTransformed.x);
                    var y_scale = vis.scale_y.invert(svgPointTransformed.y);
                    ////////////////
                    const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
                    const z_tensor = await vis.zfinder.predict(x2d_tensor)
                    // z_tensor.print()
                    const predict = await vis.Pinv.predict([x2d_tensor, z_tensor]).array()
                    // pred = clf.predict(data).arraySync()
                    const image = predict[0].map(d => d * 255)
                    update_image(vis.fake_svg, image)
                }

        })
                }
}
