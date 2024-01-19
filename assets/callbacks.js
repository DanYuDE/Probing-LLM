if (!window.dash_clientside) {
    window.dash_clientside = {};
}
console.log("callbacks.js is loaded!");

window.dash_clientside.clientside = {
    synchronizeHover: function (hoverData, graphId) {
        console.log("Current graph ID: ", graphId); // Print the current graph ID

        if (!hoverData || !graphId) {
            return null;
        }

        // Determine the target graph ID based on the source graph ID
        var targetGraphId = graphId === 'att-heatmap' ? 'block-heatmap' : 'att-heatmap';
        console.log("Target graph ID: ", targetGraphId);

        // Get the target graph DOM element
        var targetGraph = document.getElementById(targetGraphId);

        // Extract the hover information
        var hoveredLayer = hoverData.points[0].y;
        var hoveredIndex = hoverData.points[0].x;

        // Construct the new annotation to be added
        var newAnnotation = {
            x: hoveredIndex,
            y: hoveredLayer,
            text: 'Hovered',
            showarrow: true
        };

        // Use Plotly.relayout to add the new annotation
        // If you want to maintain previous annotations, you should first retrieve them
        // Here we assume you want to replace all annotations with the new one
        if (hoverData) {
            setTimeout(function () {
                window.Plotly.relayout(targetGraph, {annotations: [newAnnotation]});
            }, 500); // adjust the time (in milliseconds) as needed
        }
        return null;
    }
}