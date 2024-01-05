class MapHolder {
    constructor(data, Pinv, clf, main_svg, real_svg, fake_svg) {
        this.data = data;
        console.log(data)
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
          .range([map_height, 0])
      
        this.labels_scalar = d3.scaleOrdinal()
          .domain(d3.extent(data.label, d => d))
          .range(d3.schemeCategory10)

        
    }

    init_plots(){
        this.plot_scatters()
        this.plot_Pinv_on_click()
        // this.get2dmap()

        this.zfinder = new KNNRegressor(4);
        this.zfinder.fit(this.data.XY, this.data.z_of_XY)

    }

    get2dmap(){
        Iinv = this.Pinv.predict([tf.tensor2d(this.data.XY, [this.data.XY.length, 2]), tf.tensor2d(this.data.z_of_XY, [this.data.z_of_XY.length, 16])])
        predictions = this.clf.predict(Iinv)
        labels = predictions.argMax(1).arraySync()
        confidences = predictions.arraySync().map(d => d3.max(d))
        console.log(labels)
        console.log(confidences)
        return [labels, confidences]}

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
                update_iamge(vis.real_svg, image)

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
            .on('mousedown', function(event){this.mousedown = true})
            .on('mouseup', function(event){this.mousedown = false})
            .on("mousemove", function(event) 
            {
            if (!this.mousedown) return;
            console.log('clicking on the map')
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

            //////////////////////////////
            // // console.log(x_scale, y_scale);   
            // const z = mapholder.get_z([[x_scale, y_scale]])
            // const x2d = tf.tensor2d([x_scale, y_scale], [1, 2])
            // const z_tensor = tf.tensor2d(z, [1, 16])
            // // x2d.print()
            // // z_tensor.print()
            // const predict = mapholder.Pinv.predict([x2d, z_tensor]).arraySync()
            // // pred = clf.predict(data).arraySync()
            // const image = predict[0].map(d => d * 255)
            // update_iamge(vis.fake_svg, image)
            ////////////////
            const x2d_tensor = tf.tensor2d([x_scale, y_scale], [1, 2])
            const z_tensor = mapholder.get_z(x2d_tensor)
  
            const predict = mapholder.Pinv.predict([x2d_tensor, z_tensor]).arraySync()
            // pred = clf.predict(data).arraySync()
            const image = predict[0].map(d => d * 255)
            update_iamge(vis.fake_svg, image)
        
            // console.log(z) 
        })
    }
}

        