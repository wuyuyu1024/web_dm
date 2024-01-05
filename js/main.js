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

        // Load models and data
        const [clf, Pinv, data] = await Promise.all([
            loadModel_clf(dataset),
            loadModel_Pinv(dataset),
            loadData(dataset)
        ]).then(  
            )

        // Log the model summaries (optional)
        console.log('data' , data)
        // console.log('Model 1 Summary:', clf.summary());
        // console.log('Model 2 Summary:', Pinv.summary());

        // Now you can use model1, model2, and data together
        // For example, make predictions and process the results
        // ...
        mapholder = new MapHolder(data, Pinv, clf, main_svg, svg_real, svg_fake)
        mapholder.init_plots()

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
  })
  
  
  
  
function update_iamge(window, image, numChannels=1){
    window.selectAll(".pixel").remove()
  
    const side = Math.sqrt(image.length)
    // draw rect for image
    const width = window.attr("width")
    const height = window.attr("height")
    const cellSize_x = width / side
    const cellSize_y = height / side
  
    window.selectAll("rect")
      .data(image)
      .enter()
      .append('rect')
      .attr('x', (d, i) => (i % side) * cellSize_x) // Compute x position
      .attr('y', (d, i) => Math.floor(i / side) * cellSize_y) // Compute y position
      .attr('width', cellSize_x)
      .attr('height', cellSize_y)
      .attr('fill', function(d, i) {
        if (numChannels === 1) {
            // Grayscale: d is the intensity
            return `rgb(${d}, ${d}, ${d})`;
        } else if (numChannels === 3) {
            // RGB: every three elements represent R, G, B
            let r = image[i * numChannels];
            let g = image[i * numChannels + 1];
            let b = image[i * numChannels + 2];
            return `rgb(${r}, ${g}, ${b})`;
        } else if (numChannels === 4) {
            // RGBA: every four elements represent R, G, B, A
            let r = image[i * numChannels];
            let g = image[i * numChannels + 1];
            let b = image[i * numChannels + 2];
            let a = image[i * numChannels + 3];
            return `rgba(${r}, ${g}, ${b}, ${a})`;
        }
       }) // Set fill based on pixel intensity. if statement check the channel
      .lower() // set it to bottom
      .attr('class', 'pixel')
  
  }