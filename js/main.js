async function loadModel_clf(dataset= 'mnist') {
    path = 'data/' + dataset + '/clf_web/model.json'
    clf = await tf.loadLayersModel(path);
    return clf;
  }

async function loadModel_Pinv(dataset= 'mnist') {
    path = 'data/' + dataset + '/Pinv_web_double/model.json'
    model = await tf.loadLayersModel(path);
    return model;
}

async function loadData(dataset= 'mnist') {
    path = 'data/' + dataset + '/data.json'
    data = await d3.json(path);
    return data;
}

// main function
async function main(dataset= 'mnist') {
    try {
        // // URLs for the models and data
        // const model1Url = 'path/to/your/first/model.json';
        // const model2Url = 'path/to/your/second/model.json';
        // const dataUrl = 'path/to/your/data.json';
        // try webgpu
        await tf.setBackend('webgpu');
        await tf.ready();
        if (tf.getBackend() !== 'webgpu') {
            // Fallback to another backend
            await tf.setBackend('webgl');
        }


        // Load models and data
        const [clf, Pinv, data] = await Promise.all([
            loadModel_clf(dataset),
            loadModel_Pinv(dataset),
            loadData(dataset)
        ]).then(  
            )

        // Log the model summaries (optional)
        // console.log('data' , data)
        // console.log('Model 1 Summary:', clf.summary());
        // console.log('Model 2 Summary:', Pinv.summary());

        // Now you can use model1, model2, and data together
        // For example, make predictions and process the results
        // ...
        mapholder = new MapHolder(data, Pinv, clf, main_svg, svg_real, svg_fake)
        mapholder.init_plots()

        const map_content_dropdown = document.getElementById('map_showing_dropdown')
        map_content_dropdown.addEventListener('change', mapholder.map_showing_event)


    } catch (error) {
        console.error('Error in main execution:', error);
    }
}




/// create some frames
// main map
var main_svg = d3.select("#main_map").append("svg")
map_width = document.getElementById("main_map").offsetWidth
map_height = document.getElementById("main_map").offsetHeight 
main_svg.attr("width", map_width).attr("height", map_height).attr("id", "main_svg")

// real data
var svg_real = d3.select("#real").append("svg")
real_width = document.getElementById("real").offsetWidth 
real_height = document.getElementById("real").offsetHeight 
svg_real.attr("width", real_width).attr("height", real_height).attr("id", "real_svg")

// fake data
// real data
var svg_fake = d3.select("#fake").append("svg")
fake_width = document.getElementById("fake").offsetWidth
fake_height = document.getElementById("fake").offsetHeight
svg_fake.attr("width", fake_width).attr("height", fake_height).attr("id", "fake_svg")

//TO DO: some wigets
// tf.setBackend('webgpu').then(() => main());
main(dataset= 'mnist')



////
// main_svg.on("click", function(event) {
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
//     // console.log(x_scale, y_scale);
//     // TODO update the image
  
//   });
  
main_svg
    .on("mouseout", function(event){
    main_svg.selectAll(".moving_circle").remove()})  
    .on("mousemove", function(event) {
    // a moving circle
    // Get the SVG element's screen transformation matrix
    var CTM = main_svg.node().getScreenCTM();
    
    // Calculate the point clicked in the SVG's coordinate system
    var svgPoint = main_svg.node().createSVGPoint();
    svgPoint.x = event.clientX;
    svgPoint.y = event.clientY;
    var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
    
    main_svg.selectAll(".moving_circle").remove()
    main_svg.append("circle")
        .attr("cx", svgPointTransformed.x)
        .attr("cy", svgPointTransformed.y)
        .attr("r", 40)
        .attr("fill", "#00000000")
        .attr("stroke", 'red')
        .attr("stroke-width", 2)
        .attr("class", "moving_circle")
        // no emit event for this circle
        .attr("pointer-events", "none")
        // style for this circle: dashed
        .style("stroke-dasharray", ("3, 3"))
    
    main_svg.append("circle")
        .attr("cx", svgPointTransformed.x)
        .attr("cy", svgPointTransformed.y)
        .attr("r", 80)
        .attr("fill", "#00000000")
        .attr("stroke", 'red')
        .attr("stroke-width", 2)
        .attr("class", "moving_circle")
        // no emit event for this circle
        .attr("pointer-events", "none")
        // style for this circle: dashed
        .style("stroke-dasharray", ("6, 6"))
  })
  
  
  
  
function update_image(window, image, numChannels=1, alpha_list=null){
    let pixels = window.selectAll(".pixel").data(image);
    pixels.exit().remove();  // Exit phase
  
    const side = Math.sqrt(image.length)
    // const side = 28
    // draw rect for image
    const width = window.attr("width")
    const height = window.attr("height")
    const cellSize_x = width / side
    const cellSize_y = height / side
  
    pixels.enter().append('rect').merge(pixels) // Enter + Update phase
      .attr('x', (d, i) => (i % side) * cellSize_x) // Compute x position
      .attr('y', (d, i) => Math.floor(i / side) * cellSize_y) // Compute y position
      .attr('width', cellSize_x)
      .attr('height', cellSize_y)
      .attr('fill', function(d, i) {
            if (numChannels === 1) {
                // Grayscale: d is the intensity
                return `rgb(${d}, ${d}, ${d})`;
            } else {return d}        
                }) // Set fill based on pixel intensity. if statement check the channel
      .lower() // set it to bottom
      .attr('class', 'pixel')
      .attr('opacity', (d, i) => numChannels !== 1 ? alpha_list ? alpha_list[i] * 0.9 : 0.8 : 1)
//         .lower();
      
    //   if (numChannels !=1){
    //     if (alpha_list === null){
    //         pixels.attr("opacity", 0.8)
    //     }
    //     else{
    //         pixels.attr("opacity", (d, i) => alpha_list[i]*0.9)
    //     }
    //   }
  }

// function update_image(window, image, numChannels = 1, alpha_list = null) {
//     // Assume the image is square
//     const side = Math.sqrt(image.length / numChannels);
//     const width = window.attr("width");
//     const height = window.attr("height");
//     const cellSize_x = width / side;
//     const cellSize_y = height / side;

//     // Data join - enter, update, exit
//     // Bind the new data and remove the exiting nodes
//     let pixels = window.selectAll(".pixel").data(image);
//     pixels.exit().remove();  // Exit phase

//     // Enter phase - append new elements as needed
//     pixels.enter().append('rect')
//         .merge(pixels) // Enter + Update phase
//         .attr('x', (d, i) => (i % side) * cellSize_x)
//         .attr('y', (d, i) => Math.floor(i / side) * cellSize_y)
//         .attr('width', cellSize_x)
//         .attr('height', cellSize_y)
//         .attr('fill', d => numChannels === 1 ? `rgb(${d}, ${d}, ${d})` : d)
//         .attr('class', 'pixel')
//         .attr('opacity', (d, i) => numChannels !== 1 ? alpha_list ? alpha_list[i] * 0.9 : 0.8 : 1)
//         .lower();
// }
