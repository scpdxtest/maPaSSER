import { ContractKit } from "@wharfkit/contract"
import {APIClient} from '@wharfkit/antelope'
import React, {useState, useEffect, useRef} from "react";
import {DataTable} from 'primereact/datatable'
import {Column} from 'primereact/column';
import { Button } from "primereact/button";
import { Chart } from 'primereact/chart';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { FilterMatchMode } from 'primereact/api';
import { ProgressSpinner } from 'primereact/progressspinner';
import axios from 'axios';
import configuration from './configuration.json';

const ShowTestResults = () => {
    const [uniqueTests, SetUniqueTests] = useState([]);
    const [selectedTest, setSelectedTest] = useState('');
    const [testResults, setTestResults] = useState([]);
    const [allResults, setAllResults] = useState([]);

    const rowClass = (rowData) => {
        return {
            'p-highlight': rowData.name === selectedTest,
        };
    };

    const rowClass1 = (rowData) => {
        return {
            'p-highlight': rowData.name === resTitles[chartToShow].name,
        };
    };

    const getFiles = (data) => {
        return [...(data || [])].map((d) => {
            d.date = new Date(d.date);
            return d;
        });
    };

    const sizeBodyTemplate = (rowData) => {
        return Number(rowData.id);
    };

    const formatDate = (value) => {
        return new Date(value);
    };

    const dateBodyTemplate = (rowData) => {
        const dt = new Date(rowData.created_at)
        return dt.toISOString();
    };

    const userBodyTemplate = (rowData) => {
        return String(rowData.userid)
    }

    const resTitles = [{name: 'METEOR', color: 'rgba(75,192,192,0.4)'}, 
                    {name: 'Rouge-1.r', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-1.p', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-1.f', color: 'rgba(255, 102, 102, 0.4)'},
                    {name: 'Rouge-2.r', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-2.p', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-2.f', color: 'rgba(255, 102, 102, 0.4)'},
                    {name: 'Rouge-l.r', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-l.p', color: 'rgba(255, 102, 102, 0.4)'}, {name: 'Rouge-l.f', color: 'rgba(255, 102, 102, 0.4)'},
                    {name: 'BLEU', color: 'rgba(173, 216, 230, 0.4)'}, {name: 'Laplace Perplexity', color: 'rgba(255, 165, 0, 0.4)'}, {name: 'Lidstone Perplexity', color: 'rgba(255, 165, 0, 0.4)'},
                    {name: 'Cosine similarity', color: 'rgba(181, 101, 29, 0.4)'}, {name: 'Pearson correlation', color: 'rgba(181, 101, 29, 0.4)'}, 
                    {name: 'F1 score', color: 'rgba(147, 112, 219, 0.4)'},
                    {name: 'Bert-Score.precision', color: 'rgba(192, 112, 219, 0.4)'}, {name: 'Bert-Score.recall', color: 'rgba(203, 112, 219, 0.4)'}, {name: 'Bert-Score.f1', color: 'rgba(201, 112, 219, 0.4)'},
                    {name: 'B-RT.coherence', color: 'rgba(167, 150, 200, 0.4)'}, {name: 'B-RT.consistency', color: 'rgba(134, 124, 152, 0.4)'}, 
                    {name: 'B-RT.fluency', color: 'rgba(106, 51, 215, 0.4)'}, 
                    {name: 'B-RT.relevance', color: 'rgba(88, 80, 103, 0.4)'},
                    {name: 'B-RT.average', color: 'rgba(105, 88, 139, 0.4)'},                    
                ]

    const resultsBodyTemplate = (rowData) => {
        var ret = '';
        for (var i=0; i < 16; i++) {
            ret += resTitles[i].name + ': ' + String(rowData.results[i]) + '\n'
        }
        return ret
    }

    const contractKit = new ContractKit({        
        client: new APIClient({url: localStorage.getItem("bcEndpoint")}),        
    });

    useEffect (() => {
        loadUniqueTests();
    }, []);

    const [readingTests, setReadingTests] = useState(false);
    // Add this state at the top of your component with other state variables
    const [namesFromServer, setNamesFromServer] = useState([]);

    // Add this function to make the request
    const fetchNamesFromServer = async () => {
        try {
            console.log("Fetching names from Python server...");
            const response = await axios.get(configuration.passer.PythonNames); //'http://127.0.0.1:8088/getnames');
            
            // Store the response data in state
            setNamesFromServer(response.data);
            // console.log("Names fetched successfully:", response.data);
            
            return response.data;
        } catch (error) {
            console.error("Error fetching names:", error.message);
            return [];
        }
    };

    const loadUniqueTests = async () => {
        console.log("loadUniqueTests - start");
        setReadingTests(true);
        // const contract = contractKit.load("llmtest");
        // const cursor = (await contract).table('testtable').query();         
        // const rows = await cursor.all()
        // setAllResults(getFiles(rows));
        // let uniqueTestIds = Array.from(new Set(rows.map(row => String(row.testid))));
        // let jsonArray = uniqueTestIds.map(id => ({ name: id }));

        try {
            // Fetch names from server
            const namesData = await fetchNamesFromServer();
            
            // Check if namesData exists and has data
            if (namesData && Array.isArray(namesData)) {
                // Map the testid values to objects with name property
                const jsonArray = namesData.map(item => ({ name: item.testid }));
                
                // Set the uniqueTests state with the mapped array
                SetUniqueTests(jsonArray);
                // console.log("Unique tests set from server data:", jsonArray);
            } else {
                console.log("No test names received from server or invalid format");
                SetUniqueTests([]);
            }
        } catch (error) {
            console.error("Error in loadUniqueTests:", error);
            SetUniqueTests([]);
        } finally {
            setReadingTests(false);
            console.log("loadUniqueTests - end");
        }        

        // // run once
        // // Create array with all testid and userid combinations
        // const testsAndUsers = rows.map(row => ({
        //     testid: String(row.testid),
        //     userid: String(row.userid)
        // }));
        // // NEW CODE: Create array with unique testids only (and filter out empty values)
        // const seenTestIds = new Set();
        // const uniqueTestsAndUsers = testsAndUsers.filter(item => {
        //     // First check if either field is empty - if so, skip this item
        //     if (!item.testid || item.testid === '' || !item.userid || item.userid === '') {
        //         return false;
        //     }
            
        //     // Then check for duplicates
        //     const isDuplicate = seenTestIds.has(item.testid);
        //     if (!isDuplicate) {
        //         seenTestIds.add(item.testid);
        //         return true;
        //     }
        //     return false;
        // });
        // console.log("Unique tests and users:", uniqueTestsAndUsers);
        // // Send unique tests and users to Python server
        // const sendUniqueToPython = async () => {
        //     try {
        //         console.log("Sending unique test/user pairs to Python server...");
        //         for (const row of uniqueTestsAndUsers) {
        //             try {
        //                 await axios.post('http://127.0.0.1:8088/tests', {
        //                     userID: row.userid,
        //                     testID: row.testid
        //                 });
        //                 console.log(`Sent data for test ${row.testid} by user ${row.userid}`);
        //             } catch (error) {
        //                 console.error(`Error sending data for ${row.testid}:`, error.message);
        //             }
        //         }
        //         console.log('All unique test/user pairs processed');
        //     } catch (error) {
        //         console.error('Error in sendUniqueToPython:', error);
        //     }
        // };
        // // Call the function
        // await sendUniqueToPython();
        // console.log("sendUniqueToPython - end");

        // SetUniqueTests(jsonArray);
        // console.log("loadUniqueTests - end");
        // setReadingTests(false);
    }

    const [data, setData] = useState({});
    const options = {
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 0.001,  // Adjust this value as needed
                    callback: function(value) {
                        return parseFloat(value).toFixed(5);  // Adjust the number of decimal places as needed
                    }
                }
            }
        }
    };    
    const [chartToShow, setChartToShow] = useState(0);

    const onChartSelect = (e) => {
        const name = e.data.name;
        const index = resTitles.findIndex(item => item.name === name);
        setChartToShow(index);
        const averageValue = (!average[index] || average[index].length === 0) ? 0 : average[index];
        const sDev = (!stdDev[index] || stdDev[index].length === 0) ? 0 : stdDev[index];
        setData({
            // labels: testResults.map((_, i) => `Test ${i + 1}`),
            labels: testResults.map((r, i) => {
                const d = new Date(r.created_at);    
                return `Test ${d.toLocaleDateString()}T${d.toLocaleTimeString()}`;
            }),
            datasets: [
              {
                label: resTitles[index].name,
                data: testResults.map(test => parseFloat(test.results[index])),
                backgroundColor: resTitles[index].color,
                borderColor: resTitles[index].color.replace('0.4)', '1)'),
                borderWidth: 1,
                fill: true,
              },
              {
                label: 'Average: ' + averageValue.toFixed(4),
                data: Array(testResults.length).fill(averageValue),
                type: 'line',
                borderColor: '#000000',
                borderWidth: 0.3, // Adjust this value to make the line thinner or thicker
                fill: false,
                pointRadius: 0,
                // datalabels: {
                //     align: 'end',
                //     anchor: 'end'
                // }
              },
              {
                label: 'StdDev: ' + sDev.toFixed(4),
                data: Array(testResults.length).fill(sDev),
                type: 'line',
                borderColor: 'darkblue',
                backgroundColor: 'darkblue',
                borderWidth: 0.5, // Adjust this value to make the line thinner or thicker
                fill: false,
                pointRadius: 0,
              }
            ],
            // plugins: [ChartDataLabels],
            // options: {
            //     plugins: {
            //         datalabels: {
            //             color: '#000000',
            //             formatter: (value, context) => value.toFixed(4)
            //         }
            //     }
            // }          
        });
    }

    const [average, setAverage] = useState({});
    const [stdDev, setStdDev] = useState({});
    const [readingData, setReadingData] = useState(false);

    const onRowSelect = async (e) => {
        // console.log("onRowSelect", e.data.name);
        setReadingData(true);
        const contract = contractKit.load("llmtest");
        const cursor = (await contract).table('testtable').query({testid: e.data.name});         
        const rows = await cursor.all()
        // setAllResults(getFiles(rows));
        // console.log("++++++++++ rows", rows);

        setSelectedTest(e.data.name);
        const testID = e.data.name;
        let filteredResults = getFiles(rows).filter(result => String(result.testid) === testID);    
        setTestResults(getFiles(filteredResults));
        setReadingData(false);
// Calculate the average of each filteredResults.results[] and add it to average[]
        let averages = [];
        let stddevs = [];
        filteredResults.forEach(rowData => {
            rowData.results.forEach((result, i) => {
                if (!averages[i]) {
                    averages[i] = { sum: 0, count: 0 };
                }
                averages[i].sum += Number(result);
                averages[i].count++;

                if (!stddevs[i]) {
                    stddevs[i] = { sum: 0, sumOfSquares: 0, count: 0 };
                }
                const value = Number(result);
                stddevs[i].sum += value;
                stddevs[i].sumOfSquares += value * value;
                stddevs[i].count++;        
            });
        });
        averages = averages.map(avg => avg.count <= 0 ? 0 : avg.sum / avg.count);
        setAverage(averages);
        stddevs = stddevs.map(stddev => stddev.count <= 0 ? 0 : Math.sqrt((stddev.sumOfSquares - (stddev.sum * stddev.sum) / stddev.count) / stddev.count));
        setStdDev(stddevs);
        console.log("averages", averages, chartToShow);
        const averageValue = (!averages[chartToShow] || averages[chartToShow].length === 0) ? 0 : averages[chartToShow];
        const sDev = stddevs.length === 0 ? 0 : stddevs[chartToShow];
        console.log("filteredResults", filteredResults, getFiles(filteredResults));
        setData({
            labels: getFiles(filteredResults).map((r, i) => {
                const d = new Date(r.created_at);    
                return `Test ${d.toLocaleDateString()}T${d.toLocaleTimeString()}`;
            }),
            // labels: getFiles(filteredResults).map((_, i) => `Test ${i + 1}`),
            datasets: [
              {
                label: resTitles[chartToShow].name,
                data: getFiles(filteredResults).map(test => parseFloat(test.results[chartToShow])),
                backgroundColor: resTitles[chartToShow].color,
                borderColor: resTitles[chartToShow].color.replace('0.4)', '1)'),
                borderWidth: 1,
                fill: true,
              },
              {
                label: 'Average: ' + averageValue.toFixed(4),
                data: Array(getFiles(filteredResults).length).fill(averageValue),
                type: 'line',
                borderColor: '#000000',
                borderWidth: 0.3, // Adjust this value to make the line thinner or thicker
                fill: false,
                pointRadius: 0
              },
              {
                label: 'StdDev: ' + sDev.toFixed(4),
                data: Array(getFiles(filteredResults).length).fill(sDev),
                type: 'line',
                borderColor: 'darkblue',
                backgroundColor: 'darkblue',
                borderWidth: 0.5, // Adjust this value to make the line thinner or thicker
                fill: false,
                pointRadius: 0,
              }
            ],
        });
    };

    const calculateElapsedTime = (currentDate, previousDate) => {
        if (!previousDate) return ""; // First row has no previous date
        
        const current = new Date(currentDate);
        const previous = new Date(previousDate);
        
        // Calculate difference in seconds
        const diffInMs = current - previous;
        const diffInSeconds = Math.floor(diffInMs / 1000);
        
        return diffInSeconds;
    };

    const exportToExcel = () => {

        const XLSX = require('xlsx');
        const myData = [];

        for (var i=0; i<testResults.length; i++) {
            const dt = new Date(testResults[i].created_at);
            var tmpRes = {id: Number(testResults[i].id), 
                            date: String(testResults[i].created_at),
                            userID: String(testResults[i].userid),
                            testID: String(testResults[i].testid),
                            Description: testResults[i].description,
                        }
            for (var j=0; j < 24; j++) {
            // Convert to number, but handle non-numeric values gracefully
                const value = testResults[i].results[j];
                const numValue = Number(value);
                
                // Use number if valid, otherwise use original value
                tmpRes[resTitles[j].name] = !isNaN(numValue) ? numValue : value;
                // tmpRes[resTitles[j].name] = String(testResults[i].results[j]);
            }           
            if (i > 0) {tmpRes["Elapsed time"] = calculateElapsedTime(testResults[i].created_at, testResults[i-1].created_at)} else 
            {tmpRes["Elapsed time"] = ""}
            myData.push(tmpRes);
        }

        // Create a new workbook
        const wb = XLSX.utils.book_new();

        // Convert the array to a worksheet
        const ws = XLSX.utils.json_to_sheet(myData);

        // Add the worksheet to the workbook
        XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');

        // Write the workbook to a file
        XLSX.writeFile(wb, 'output.xlsx');  
    }

    const [filters, setFilters] = useState({
        global: { value: null, matchMode: FilterMatchMode.CONTAINS },
        name: { value: null, matchMode: FilterMatchMode.STARTS_WITH },
    });

    const chartRef = useRef();

    const downloadChart = () => {
        console.log('chartRef', chartRef.current.chartInstance);
        if (chartRef.current) {
            const canvas = chartRef.current.getCanvas();
            // const ctx = chartRef.current.getCanvas().getContext('2d');
            // console.log('canvas', canvas, ctx);
            // ctx.fillStyle = 'white';
            // ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            const base64Image = canvas.toDataURL("image/png");
            const link = document.createElement('a');
            link.href = base64Image;
            link.download = 'chart.png';
            link.click();
        }
    };

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            <div style={{ width: '12%'}}>
                <h3>Unique tests</h3>
                {readingTests ? (
                    <div className="p-d-flex p-flex-column p-ai-center" style={{ textAlign: 'center', padding: '1rem' }}>
                        <ProgressSpinner style={{ width: '50px', height: '50px' }} />
                        <div style={{ marginTop: '0.5rem' }}>Reading from blockchain</div>
                    </div>
                ) : (
                    <DataTable style={{width: '90%'}} value={uniqueTests}   
                        onRowSelect={onRowSelect} selectionMode="single" selection={selectedTest}
                        rowClassName={rowClass} 
                        filters={filters} filterDisplay="row"
                        size={"small"} showGridlines stripedRows>
                        <Column field="name" header="Name" filter filterPlaceholder="By name" ></Column>
                    </DataTable>
                )}
            </div>

            <div style={{ width: '88%'}}>
                <h3>Test results</h3>
                <Button label="Export to Excel" style={{marginBottom: '10px'}} onClick={exportToExcel}/>
                <Button style={{marginBottom: '10px', marginLeft: '10px'}} label="Download chart" onClick={downloadChart} />
                <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
                    <div style={{ width: '20%'}}>
                        {readingData ? (
                            <div className="p-d-flex p-flex-column p-ai-center" style={{ textAlign: 'center', padding: '1rem' }}>
                                <ProgressSpinner style={{ width: '50px', height: '50px' }} />
                                <div style={{ marginTop: '0.5rem' }}>Reading from blockchain</div>
                            </div>
                        ) : (
                            <DataTable style={{width: '90%'}} value={resTitles}   
                                onRowSelect={onChartSelect} selectionMode="single" selection={chartToShow}
                                rowClassName={rowClass1}
                                size={"small"} showGridlines stripedRows>
                                <Column field="name" header="Name"></Column>
                            </DataTable>
                        )}
                    </div>
                    <div style={{ width: '100%'}}>
                        <div className="chart-container" style={{ overflowX: 'auto', width: '100%' }}>
                            <Chart ref={chartRef} type="bar" data={data} options={options} style={{ minWidth: '2000px' }} />
                        </div>

                            {/* <Chart type="bar" data={data} options={options} /> */}
                    </div>
                </div>
                {average.length > 0 ? (
                    <div style={{ 
                        width: '40%', 
                        border: '1px solid black', 
                        padding: '10px', 
                        margin: '0 auto', 
                        lineHeight: '1.5', 
                        textAlign: 'center' 
                    }}>
                        <h3>Averages</h3>
                        {average.map((average, i) => (
                            <p key={i} style={{ margin: '0.5em 0' }}><strong>{resTitles[i].name}:</strong> {average}</p>
                        ))}
                    </div>
                ) : null}                
                <DataTable style={{width: '90%'}} value={testResults}   
                    selectionMode="single"
                    rowClassName={rowClass}
                    size={"small"} showGridlines stripedRows>

                    <Column field="id" header="ID" dataType="number" body={sizeBodyTemplate}></Column>
                    <Column field="created_at" header="Date" dataType="date" body={dateBodyTemplate} style={{width: '20%'}}/>
                    <Column field="userid" header="userID" dataType="string" body={userBodyTemplate} style={{width: '10%'}}/>
                    <Column field="results" header="Results" dataType="string" body={resultsBodyTemplate} style={{width: '40%'}}/>
                    <Column field="description" header="Description" dataType="string" style={{width: '40%'}}/>
                </DataTable>
            </div>
        </div>

    );

}

export default ShowTestResults;