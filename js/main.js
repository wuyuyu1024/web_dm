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


// Function to set up the TensorFlow.js backend
async function setUpTFBackend() {
    try {
        // Explicitly set backend to 'webgpu'
        await tf.setBackend('webgpu');

        // Wait for the backend to be ready
        await tf.ready();

        // Check if the backend is successfully set to 'webgpu'
        if (tf.getBackend() !== 'webgpu') {
            console.warn("WebGPU is not available. Falling back to 'webgl'.");
            await tf.setBackend('webgl');
            await tf.ready();
        }

        console.log(`Using TensorFlow.js with backend: ${tf.getBackend()}`);
    } catch (error) {
        console.error('Error setting up TensorFlow.js backend:', error);
    }
}


// main function
async function main(dataset= 'mnist') {
    try {
        // // URLs for the models and data
        // const model1Url = 'path/to/your/first/model.json';
        // const model2Url = 'path/to/your/second/model.json';
        // const dataUrl = 'path/to/your/data.json';
        // try webgpu
        await setUpTFBackend();
        await console.log(`Using TensorFlow.js with backend: ${tf.getBackend()}`);
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
        scale_x = mapholder.scale_x
        scale_y = mapholder.scale_y
        mapholder.init_plots()
        
        // dropdown for map content
        const map_content_dropdown = document.getElementById('map_showing_dropdown')
        map_content_dropdown.addEventListener('change', mapholder.map_showing_event)
        
        // checkbox for interaction
        const interaction_checkbox = document.getElementById('checkbox_adjust')
        interaction_checkbox.addEventListener('change', mapholder.hold_checkbox_event)

        // slider for sigma
        const sigma_slider = document.getElementById('slider_radius')
        sigma_slider.addEventListener('change', mapholder.radius_slider_event)

        // slider for factor
        // const factor_slider = document.getElementById('slider_factor')
        // factor_slider.addEventListener('change', mapholder.factor_slider_event)

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
.on('mousemove', (event)=>{console.log('mouse move detect')})
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


// Function to resize SVGs/////////////////////////////
// function resizeSVGs() {
//     // Resize main map
//     map_width = document.getElementById("main_map").offsetWidth;
//     map_height = document.getElementById("main_map").offsetHeight;
//     d3.select("#main_svg").attr("width", map_width).attr("height", map_height);

//     // Resize real data SVG
//     real_width = document.getElementById("real").offsetWidth;
//     real_height = document.getElementById("real").offsetHeight;
//     d3.select("#real_svg").attr("width", real_width).attr("height", real_height);

//     // Resize fake data SVG
//     fake_width = document.getElementById("fake").offsetWidth;
//     fake_height = document.getElementById("fake").offsetHeight;
//     d3.select("#fake_svg").attr("width", fake_width).attr("height", fake_height);
// }

// // Initial resize
// resizeSVGs();

// // Resize event listener
// window.addEventListener("resize", resizeSVGs);
/////////////////////////////////////////////

// Your existing code for appending SVGs
// (Ensure this runs before the initial call to resizeSVGs)
// var main_svg = d3.select("#main_map").append("svg").attr("id", "main_svg");
// var svg_real = d3.select("#real").append("svg").attr("id", "real_svg");
// var svg_fake = d3.select("#fake").append("svg").attr("id", "fake_svg");


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
  

// Cache elements and calculations that don't change
// const adjust_checkbox = document.getElementById('checkbox_adjust');
// const CTM = main_svg.node().getScreenCTM();
// const svgPoint = main_svg.node().createSVGPoint();
// let movingCircle = null;
// let movingCircleLabel = null;

// main_svg.on("mouseout", function(event){
//     if (movingCircle) movingCircle.remove();
//     if (movingCircleLabel) movingCircleLabel.remove();
// })  
// .on("mousemove", function(event) {
//     // Early return if checkbox is not checked
//     if (!adjust_checkbox.checked) return;
//     // console.log('moving')
//     svgPoint.x = event.clientX;
//     svgPoint.y = event.clientY;
//     var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());

//     // Update or create the moving circle
//     if (movingCircle == null) {
//         // console.log('create moving circle')
//         movingCircle = main_svg.append("circle")
//             .attr("fill", "#00000000")
//             .attr("stroke", 'red')
//             .attr("stroke-width", 2)
//             .attr("class", "moving_circle")
//             .attr("pointer-events", "none")
//             .style("stroke-dasharray", ("3, 5"));
//     }
//     movingCircle
//         .attr("cx", svgPointTransformed.x)
//         .attr("cy", svgPointTransformed.y)
//         .attr("r", 40);

//     // Update or create the label for moving circle
//     if (!movingCircleLabel) {
//         movingCircleLabel = main_svg.append("text")
//             .attr("class", "label_for_moving_circle")
//             .attr("fill", "red")
//             .attr("font-size", "12px")
//             .attr("text-anchor", "start")
//             .text("1σ");
//     }
//     movingCircleLabel
//         .attr("x", svgPointTransformed.x - 2)
//         .attr("y", svgPointTransformed.y + 55); // Adjusted for the radius + some padding
// });

//  my old
// var scale_x
// var scale_y

// a moving circle
// Get the SVG element's screen transformation matrix
var CTM = main_svg.node().getScreenCTM();
// Calculate the point clicked in the SVG's coordinate system
var svgPoint = main_svg.node().createSVGPoint();
let value_slider_radius = document.getElementById('slider_radius').value
// inverse the scale
// let value_on_scale = scale_x(value_slider_radius)
// console.log(value_on_scale)

// var radius = value_on_scale

// var moving_circle = main_svg.append("circle")
//     .attr("r", radius)
//     .attr("fill", "#00000000")
//     .attr("stroke", 'red')
//     .attr("stroke-width", 2)
//     .attr("class", "moving_circle")
//     // no emit event for this circle
//     .attr("pointer-events", "none")
//     // style for this circle: dashed
//     .style("stroke-dasharray", ("3, 5"))
//     // .upper()

// var label_cirle = main_svg.append("text")
//     .attr("class", "label_for_moving_circle")
//     .attr("fill", "red")
//     .attr("font-size", "12px")
//     .attr("text-anchor", "start")
//     .text("1σ")
    

// main_svg
// //     .on("mouseout", function(event){
// //     main_svg.selectAll(".moving_circle")
// //         .attr("cx", -30)
// //         .attr("cy", -30)
// //     main_svg.selectAll(".label_for_moving_circle")
// //         .attr("x", 0)
// //         .attr("y", 0)
// // })  
//     .on("mousemove", function(event) {
//         console.log('move out side')})
            // read the checkbox
        // let adjust_checkbox = document.getElementById('checkbox_adjust')
        // if (adjust_checkbox.checked === false) {return}
        // svgPoint.x = event.clientX;
        // svgPoint.y = event.clientY;
        // var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());
        // moving_circle.raise()
        //     .attr("cx", svgPointTransformed.x)
        //     .attr("cy", svgPointTransformed.y)


        // label_cirle.raise()
        //     .attr("x", svgPointTransformed.x -2)
        //     .attr("y", svgPointTransformed.y + radius + 15)

    // main_svg.select('.label_for_moving_circle').remove()
    
    // })
    
    
    // main_svg.append("circle")
    //     .attr("cx", svgPointTransformed.x)
    //     .attr("cy", svgPointTransformed.y)
    //     .attr("r", 80)
    //     .attr("fill", "#00000000")
    //     .attr("stroke", 'red')
    //     .attr("stroke-width", 2)
    //     .attr("class", "moving_circle")
    //     // no emit event for this circle
    //     .attr("pointer-events", "none")
    //     // style for this circle: dashed
    //     .style("stroke-dasharray", ("6, 6"))

  
  
  
  
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
      .attr("pointer-events", "none")
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
