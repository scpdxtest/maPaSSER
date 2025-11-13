import { ContractKit } from "@wharfkit/contract"
import {APIClient} from '@wharfkit/antelope'
import React, {useState, useEffect, useRef} from "react";
import {DataTable} from 'primereact/datatable'
import {Column} from 'primereact/column';
import { Button } from "primereact/button";
import { Chart } from 'primereact/chart';
import { ProgressSpinner } from 'primereact/progressspinner';

const ShowTimeTestResults = () => {
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

    const resTitles = [{name: 'Evaluation time'}, {name: 'Evaluation count'}, {name: 'Load duration time'}, 
                        {name: 'Prompt evaluation count'},
                        {name: 'Prompt evaluation duration'}, {name: 'Total duration'}, {name: 'Tokens per second'}]

    const resultsBodyTemplate = (rowData) => {
        var ret = '';
        for (var i=0; i < 6; i++) {
            ret += resTitles[i].name + ': ' + String(rowData.results[i]) + '\n'
        }
        ret += resTitles[6].name + ': ' + (Number(rowData.results[1]) / Number(rowData.results[0]) * 1e9).toString()
        return ret
    }

    const contractKit = new ContractKit({        
        client: new APIClient({url: localStorage.getItem("bcEndpoint")}),        
    });

    useEffect (() => {
        loadUniqueTests();
    }, []);

    const [readingTests, setReadingTests] = useState(false);

    const loadUniqueTests = async () => {
        // console.log("loadUniqueTests");
        setReadingTests(true);
        const contract = contractKit.load("llmtest");
        const cursor = (await contract).table('timetable').query();
            
        const rows = await cursor.all()
        setAllResults(getFiles(rows));
        let uniqueTestIds = Array.from(new Set(rows.map(row => String(row.testid))));
        let jsonArray = uniqueTestIds.map(id => ({ name: id }));
        SetUniqueTests(jsonArray);
        setReadingTests(false);
    }

    const [data, setData] = useState({});
    const options = {
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 0.1,  // Adjust this value as needed
                    callback: function(value) {
                        return parseFloat(value).toFixed(0);  // Adjust the number of decimal places as needed
                    }
                }
            }
        }
    };    

    const onRowSelect = async (e) => {
        setSelectedTest(e.data.name);
        const testID = e.data.name;
        var filteredResults = allResults.filter(result => String(result.testid) === testID);    
        filteredResults = filteredResults.map(rowData => ({
            ...rowData,
            results: [...rowData.results, (Number(rowData.results[1]) / Number(rowData.results[0]) * 1e9)]
        }));        
        // console.log("onRowSelect", filteredResults);    
        setTestResults(getFiles(filteredResults));
        setData({
            // labels: getFiles(filteredResults).map((_, i) => `Test ${i + 1}`),
            labels: getFiles(filteredResults).map((r, i) => {
                const d = new Date(r.created_at);    
                return `Test ${d.toLocaleDateString()}T${d.toLocaleTimeString()}`;
            }),
            datasets: [
              {
                label: resTitles[chartToShow].name,
                data: getFiles(filteredResults).map(test => parseFloat(test.results[chartToShow])),
                backgroundColor: 'rgba(75,192,192,0.4)',
                borderColor: 'rgba(75,192,192,1)',
                borderWidth: 1,
                fill: false,
                type: 'line',
            },
            ],
        });
    };

    const [chartToShow, setChartToShow] = useState(0);

    const onChartSelect = (e) => {
        const name = e.data.name;
        const index = resTitles.findIndex(item => item.name === name);
        // console.log("onChartSelect", index, e.data.name);
        setChartToShow(index);
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
                backgroundColor: 'rgba(75,192,192,0.4)',
                borderColor: 'rgba(75,192,192,1)',
                borderWidth: 1,
                fill: false,
                type: 'line',
            },
            ],
        });

    }

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
            for (var j=0; j < 6; j++) {
                tmpRes[resTitles[j].name] = String(testResults[i].results[j]);
            }           
            tmpRes[resTitles[6].name] = (Number(testResults[i].results[1]) / Number(testResults[i].results[0]) * 1e9).toString()            
            myData.push(tmpRes);
        }

        // Create a new workbook
        const wb = XLSX.utils.book_new();

        // Convert the array to a worksheet
        const ws = XLSX.utils.json_to_sheet(myData);

        // Add the worksheet to the workbook
        XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');

        // Write the workbook to a file
        XLSX.writeFile(wb, 'times_output.xlsx');  
    }
    
    const chartRef = useRef();

    const downloadChart = () => {
        console.log('chartRef', chartRef.current.chartInstance);
        if (chartRef.current) {
            const canvas = chartRef.current.getCanvas();
            const base64Image = canvas.toDataURL("image/png");
            const link = document.createElement('a');
            link.href = base64Image;
            link.download = 'chart.png';
            link.click();
        }
    };

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            <div style={{ width: '10%'}}>
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
                    size={"small"} showGridlines stripedRows>
                        <Column field="name" header="Name"></Column>
                    </DataTable>
                )}
            </div>

            <div style={{ width: '90%'}}>
                <h3>Time Test results</h3>
                <Button label="Export to Excel" style={{marginBottom: '10px'}} onClick={exportToExcel}/>
                <Button style={{marginBottom: '10px', marginLeft: '10px'}} label="Download chart" onClick={downloadChart} />
                <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
                    <div style={{ width: '20%'}}>
                        <DataTable style={{width: '90%'}} value={resTitles}   
                            onRowSelect={onChartSelect} selectionMode="single" selection={chartToShow}
                            rowClassName={rowClass1}
                            size={"small"} showGridlines stripedRows>
                            <Column field="name" header="Name"></Column>
                        </DataTable>
                    </div>
                    <div style={{ width: '100%'}}>
                        <div className="chart-container" style={{ overflowX: 'auto', width: '100%' }}>
                            <Chart ref={chartRef} type="bar" data={data} options={options} style={{ minWidth: '2000px' }} />
                        </div>
                    </div>
                </div>
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

export default ShowTimeTestResults;