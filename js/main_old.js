
// import * as tf from '@tensorflow/tfjs';
var clf 
// Load the model.
async function loadModel() {
  clf = await tf.loadLayersModel('data/mnist/Pinv_web_double/model.json');
  
  return clf;
}

clf = loadModel().then (function (clf){
// console.log(clf.summary());
// dummy random data 784
let data = tf.randomNormal([1, 18]);
pred = clf.predict([tf.randomNormal([1, 2]), tf.randomNormal([1, 16])]).arraySync()
// pred = clf.predict(data).arraySync()
image = pred[0].map(d => d * 255)
console.log(image)
update_iamge(main_svg, image)
}
)
// console.log(pred)

// console.log(clf.summary());


// Variables for the visualization instances
let main_svg = d3.select("#main_map").append("svg")
map_width = document.getElementById("main_map").offsetWidth
map_height = document.getElementById("main_map").offsetHeight
main_svg.attr("width", map_width).attr("height", map_height).attr("id", "main_svg")
      // .on("mousemove", function(event){ console.log(event.layerX, event.layerY) 
      //   main_svg.append("circle")
      //   .attr("cx", event.layerX)
      //   .attr("cy", event.layerY)
      //   .attr("r", 10)
      //   .attr("fill", null)
      //   .attr("stroke", "white")
      //   .attr("stroke-width", 2)

      // })

// main_svg.append("rect")
//   .attr("width", "100%")
//   .attr("height", "100%")
//   .attr("fill", "red")
//   .attr("opacity", 0.05)
  

let svg_real = d3.select("#real").append("svg")
real_width = document.getElementById("real").offsetWidth
real_height = document.getElementById("real").offsetHeight
svg_real.attr("width", real_width).attr("height", real_height).attr("id", "real_svg")


var scale_x
var scale_y

// Start application by loading the data
loadData();

function loadData() {
  d3.json("data/mnist/data.json").then((jsonData) => {
    console.log(jsonData);
    // prepare data
    // let data = prepareDataForStudents(jsonData);

    // console.log("data loaded ");
    // console.log(data);

    // TO-DO (Activity I): instantiate visualization objects
    //stackchart = new StackedAreaChart('stacked-area-chart', data.layers);
    //timeline = new Timeline('timeline', data.years);

    // TO-DO (Activity I):  init visualizations
    //stackchart.initVis();
    //timeline.initVis();

    // one: scatter plot
    plot_scatters(jsonData)
  });
}



function brushed() {
  // TO-DO: React to 'brushed' event
  console.log("brushed");
  let selectionRange = d3.brushSelection(d3.select(".brush").node());
  let selectedYears = selectionRange.map(timeline.x.invert);
  // console.log(selectedYears);
  filterd_data = stackchart.data.filter(function (d) {
    return d.Year >= selectedYears[0] && d.Year <= selectedYears[1];
  });
  // console.log("Filtered data")
  // console.log(filterd_data)

  // newStackData = stackchart.stackingData(filterd_data);
  stackchart.wrangleData(filterd_data);
  
}


function plot_scatters(data){
  // console.log(data)

  padding = 0.05
  padding_x = (d3.max(data.X2d, d => d[0]) - d3.min(data.X2d, d => d[0])) * padding
  padding_y = (d3.max(data.X2d, d => d[1]) - d3.min(data.X2d, d => d[1])) * padding
  // console.log(padding_x)
  // console.log(padding_y)

  scale_x = d3.scaleLinear()
    .domain([d3.min(data.X2d, d => d[0]) - padding_x, d3.max(data.X2d, d => d[0]) + padding_x])
    .range([0, map_width])

  scale_y = d3.scaleLinear()
    .domain([d3.min(data.X2d, d => d[1]) - padding_y, d3.max(data.X2d, d => d[1]) + padding_y])
    .range([map_height, 0])

  labels_scalar = d3.scaleOrdinal()
    .domain(d3.extent(data.label, d => d))
    .range(d3.schemeCategory10)

  scatters = main_svg.selectAll("circle")
    .data(data.X2d)
    .enter()
    .append("circle")
    .attr("cx", d => scale_x(d[0]))
    .attr("cy", d => scale_y(d[1]))
    .attr("r", 4)
    .attr("fill", (d, i) => labels_scalar(data.label[i]))
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
      main_svg.selectAll(".selected_circle").remove()
      
      index = data.X2d.indexOf(x2d)
      console.log(index)
      image = data.X[index].map(d => d * 255)
      update_iamge(svg_real, image)

      main_svg.append("circle")
              .attr("cx", scale_x(x2d[0]))
              .attr("cy", scale_y(x2d[1]))
              .attr("r", d3.select(this).attr("r"))
              .attr("fill", d3.select(this).attr("fill"))
              .attr("stroke", "red")
              .attr("stroke-width", 2)
              .attr('class', 'selected_circle')
              .attr("pointer-events", "none")

    })
}

// main_svg.on("click", function(event){
//   console.log(event)
//   x = event.clientX
//   y = event.clientY
//   x_scale = scale_x.invert(x)
//   y_scale = scale_y.invert(y)
//   console.log(x_scale)
//   console.log(y_scale)
//   console.log('------------')
//   main_svg.append("circle")
//     .attr("cx", x)
//     .attr("cy", y)
//     .attr("r", 10)
//     .attr("fill", 'red')

//   }
// )

main_svg.on("click", function(event) {
  // main_svg.selectAll(".selected_circle").remove()
  // Get the SVG element's screen transformation matrix
  var CTM = main_svg.node().getScreenCTM();

  // Calculate the point clicked in the SVG's coordinate system
  var svgPoint = main_svg.node().createSVGPoint();
  svgPoint.x = event.clientX;
  svgPoint.y = event.clientY;
  var svgPointTransformed = svgPoint.matrixTransform(CTM.inverse());

  // Now svgPointTransformed.x and svgPointTransformed.y are the correct coordinates
  console.log(svgPointTransformed.x, svgPointTransformed.y);

  // Use your scales to invert the coordinates
  var x_scale = scale_x.invert(svgPointTransformed.x);
  var y_scale = scale_y.invert(svgPointTransformed.y);
  console.log(x_scale, y_scale);
  // TODO update the image

});

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

// const modelDataHandler = new ModelDataHandler(
//   'path/to/your/first/model.json',
//   'path/to/your/second/model.json',
//   'path/to/your/data.json'
// );

// modelDataHandler.initialize().then(() => {
//   console.log("Models and data loaded successfully");

//   // Perform operations with the loaded models and data
//   modelDataHandler.performOperation()
//       .then(() => {
//           console.log("Operation performed successfully");
//       })
//       .catch(error => {
//           console.error("Error in performing operation:", error);
//       });
// })
// .catch(error => {
//   console.error("Error in initialization:", error);
// });