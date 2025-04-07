/**
 * Initializes a Vanta.js animated background using the "BIRDS" effect
 * once the DOM content has fully loaded.
 */
document.addEventListener("DOMContentLoaded", function () {
  VANTA.BIRDS({
    el: "#backgroundJS",         // Element selector for the canvas background effect
    mouseControls: true,         // Enables interaction with the animation via mouse movement
    touchControls: true,         // Enables interaction via touch (mobile devices)
    gyroControls: false,         // Disables gyroscope-based interaction
    minHeight: 200.00,           // Minimum canvas height
    minWidth: 200.00,            // Minimum canvas width
    scale: 1.00,                 // Scale factor for desktop
    scaleMobile: 1.00,           // Scale factor for mobile devices
    birdSize: 1.50,              // Adjusts the size of birds
    wingSpan: 32.00,             // Determines how wide the wings are
    speedLimit: 10.00,           // Maximum speed the birds can move
    quantity: 2.00               // Number of birds on screen
  });
});
