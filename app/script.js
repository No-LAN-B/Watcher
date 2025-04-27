// Note: This project is partially a learning exercise so commenting why / how something works will be very common as I use those comments to learn and remember.

// Wait for the HTML document to be fully loaded before running chart code
document.addEventListener('DOMContentLoaded', (event) => {
    console.log("DOM fully loaded and parsed");
     
    // --- Chart Setup ---
    const chartElement = document.getElementById('chart');
    console.log("Chart element:", chartElement); 

    // Check if the element was actually found
    if (!chartElement) {
        console.error("Error: Could not find chart element with ID 'chart'");
        return; // Stop if element doesn't exist
    }

    // Add this log BEFORE creating the chart:
    console.log("LightweightCharts library:", LightweightCharts); 

    let chart = null; // Define chart variable outside try block
    try {
        chart = LightweightCharts.createChart(chartElement, {
            // Using fixed dimensions - keep this for now
            width: 600, 
            height: 400, 
            layout: {
                backgroundColor: '#ffffff',
                textColor: '#333',
            },
            grid: {
                vertLines: { color: '#f0f0f0' },
                horzLines: { color: '#f0f0f0' },
            },
            timeScale: { 
                timeVisible: true,
                secondsVisible: false,
            }
        });
        // Add this log AFTER creating the chart:
        console.log("Chart object created:", chart); 
    } catch (error) {
        console.error("Error creating chart:", error);
        return; // Stop if chart creation fails
    }


    // Example: Add a Candlestick series
    // Ensure 'chart' object exists before calling this
    let candlestickSeries = null; 
    try {
        // --- CORRECTED SYNTAX for adding series ---
        // Use addSeries() and pass the series type object
        candlestickSeries = chart.addSeries(LightweightCharts.CandlestickSeries, { 
          // Put your options back inside this object
          upColor: '#26a69a', // Green
          downColor: '#ef5350', // Red
          borderDownColor: '#ef5350',
          borderUpColor: '#26a69a',
          wickDownColor: '#ef5350',
          wickUpColor: '#26a69a',
        });
        // -----------------------------------------

        console.log("Candlestick series added successfully.");
    } catch (error) {
        // This error should hopefully be gone now!
        console.error("Error adding candlestick series:", error); 
    }


    // --- Placeholder Data Loading ---
    const sampleData = [
        { time: '2025-04-10', open: 170, high: 172, low: 169, close: 171 },
        { time: '2025-04-11', open: 171, high: 173, low: 170, close: 172 },
        { time: '2025-04-14', open: 172, high: 172.5, low: 170.5, close: 171.5 },
        { time: '2025-04-15', open: 171.5, high: 174, low: 171, close: 173.5 },
        { time: '2025-04-16', open: 173.5, high: 175, low: 173, close: 174 },
        { time: '2025-04-17', open: 174, high: 176, low: 173.5, close: 175.5 },
    ];

    if (candlestickSeries) {
        candlestickSeries.setData(sampleData);
    } else {
        console.log("Skipping setData because candlestickSeries is null.");
    }


    // --- Function for Loading Symbol Data ---
    // Ensure this global function is accessible (defined outside DOMContentLoaded if needed elsewhere, but fine here for now)
    // Make it available globally if onclick attributes need it
    window.loadSymbol = async function(symbol) { 
        console.log(`Fetching data for symbol: ${symbol} from backend...`);
        const apiUrl = `http://127.0.0.1:5000/api/stockdata?symbol=${symbol}`; 

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                const errorData = await response.json();
                console.error(`Error fetching data: ${response.status} ${response.statusText}`, errorData);
                alert(`Error loading data for ${symbol}: ${errorData.error || response.statusText}`);
                if (candlestickSeries) candlestickSeries.setData([]);
                return; 
            }
            const fetchedData = await response.json();
            if (!Array.isArray(fetchedData)) {
                 console.log("Received non-array data from backend:", fetchedData);
                 alert(`Invalid data format received for ${symbol}. Expected an array.`);
                 if (candlestickSeries) candlestickSeries.setData([]);
                 return;
            }
            if (fetchedData.length === 0) {
                 console.log("Received empty data array from backend.");
                 if (candlestickSeries) candlestickSeries.setData([]);
                 return;
            }
            console.log(`Data received for ${symbol}, setting chart data.`);
            if (candlestickSeries) {
                 candlestickSeries.setData(fetchedData);
                 if (chart) chart.timeScale().fitContent(); 
            } else {
                console.error("Cannot set data because candlestickSeries was not created successfully.");
            }
        } catch (error) {
            console.error("Network error or failed fetch:", error);
            alert(`Failed to connect to the backend to load data for ${symbol}. Is the Flask server running?`);
            if (candlestickSeries) candlestickSeries.setData([]); 
        }
    } // End of loadSymbol function definition

    // Adjust chart size on window resize
    window.addEventListener('resize', () => {
        if (chart) {
            // Still using fixed initial size, resize might need more logic if dynamic
             chart.resize(chartElement.clientWidth > 0 ? chartElement.clientWidth : 600, 
                          chartElement.clientHeight > 0 ? chartElement.clientHeight : 400); 
        }
    });

    // Skipping initial loadSymbol call for now
    console.log("Skipping initial loadSymbol call.");

}); // End of DOMContentLoaded listener