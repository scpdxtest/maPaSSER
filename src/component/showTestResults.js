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
import { InputText } from 'primereact/inputtext';
import axios from 'axios';
import configuration from './configuration.json';

const ShowTestResults = () => {
    const [uniqueTests, SetUniqueTests] = useState([]);
    const [selectedTest, setSelectedTest] = useState('');
    const [testResults, setTestResults] = useState([]);
    const [allResults, setAllResults] = useState([]);
    const [filterPrefix, setFilterPrefix] = useState('');
    const [dataSource, setDataSource] = useState('blockchain'); // 'blockchain' or 'mongodb'

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
        return String(rowData.userid || rowData.creator)
    }

    // Updated metrics titles - Base NLP metrics (24) + MultiAgent metrics (18)
    const resTitles = [
        // Base NLP Metrics (0-23)
        {name: 'METEOR', color: 'rgba(75,192,192,0.4)'}, 
        {name: 'BLEU', color: 'rgba(173, 216, 230, 0.4)'}, 
        {name: 'Rouge-1.r', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-1.p', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-1.f', color: 'rgba(255, 102, 102, 0.4)'},
        {name: 'Rouge-2.r', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-2.p', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-2.f', color: 'rgba(255, 102, 102, 0.4)'},
        {name: 'Rouge-l.r', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-l.p', color: 'rgba(255, 102, 102, 0.4)'}, 
        {name: 'Rouge-l.f', color: 'rgba(255, 102, 102, 0.4)'},
        {name: 'Laplace Perplexity', color: 'rgba(255, 165, 0, 0.4)'}, 
        {name: 'Lidstone Perplexity', color: 'rgba(255, 165, 0, 0.4)'},
        {name: 'Cosine similarity', color: 'rgba(181, 101, 29, 0.4)'}, 
        {name: 'Pearson correlation', color: 'rgba(181, 101, 29, 0.4)'}, 
        {name: 'F1 score', color: 'rgba(147, 112, 219, 0.4)'},
        {name: 'Bert-Score.precision', color: 'rgba(192, 112, 219, 0.4)'}, 
        {name: 'Bert-Score.recall', color: 'rgba(203, 112, 219, 0.4)'}, 
        {name: 'Bert-Score.f1', color: 'rgba(201, 112, 219, 0.4)'},
        {name: 'B-RT.coherence', color: 'rgba(167, 150, 200, 0.4)'}, 
        {name: 'B-RT.consistency', color: 'rgba(134, 124, 152, 0.4)'}, 
        {name: 'B-RT.fluency', color: 'rgba(106, 51, 215, 0.4)'}, 
        {name: 'B-RT.relevance', color: 'rgba(88, 80, 103, 0.4)'},
        {name: 'B-RT.average', color: 'rgba(105, 88, 139, 0.4)'},
        
        // MultiAgent Performance Metrics (24-41) - 18 additional metrics
        {name: 'Retrieval Time (ms)', color: 'rgba(54, 162, 235, 0.4)'},      // 24
        {name: 'Generation Time (ms)', color: 'rgba(255, 206, 86, 0.4)'},     // 25
        {name: 'Consensus Time (ms)', color: 'rgba(75, 192, 192, 0.4)'},      // 26
        {name: 'End-to-End Time (ms)', color: 'rgba(153, 102, 255, 0.4)'},    // 27
        {name: 'Prompt Tokens', color: 'rgba(255, 159, 64, 0.4)'},            // 28
        {name: 'Completion Tokens', color: 'rgba(255, 99, 132, 0.4)'},        // 29
        {name: 'Total Tokens', color: 'rgba(201, 203, 207, 0.4)'},            // 30
        {name: 'Messages Total', color: 'rgba(255, 205, 86, 0.4)'},           // 31
        {name: 'Turns Total', color: 'rgba(54, 162, 235, 0.4)'},              // 32
        {name: 'Agents Participated', color: 'rgba(75, 192, 192, 0.4)'},      // 33
        {name: 'Consensus Rounds', color: 'rgba(153, 102, 255, 0.4)'},        // 34
        {name: 'Disagreement Rate', color: 'rgba(255, 159, 64, 0.4)'},        // 35
        {name: 'Consensus Score (%)', color: 'rgba(46, 204, 113, 0.4)'},      // 36
        {name: 'Processing Time (s)', color: 'rgba(52, 152, 219, 0.4)'},      // 37
        {name: 'Number of Agents', color: 'rgba(155, 89, 182, 0.4)'},         // 38
        {name: 'Strategy Type', color: 'rgba(241, 196, 15, 0.4)'},            // 39
        {name: 'RAG Enabled', color: 'rgba(230, 126, 34, 0.4)'},              // 40
        {name: 'Docs Retrieved', color: 'rgba(231, 76, 60, 0.4)'},            // 41
    ];

    const resultsBodyTemplate = (rowData) => {
        var ret = '';
        const totalMetrics = rowData.results.length;
        
        // Show all available metrics
        for (var i = 0; i < totalMetrics; i++) {
            if (i < resTitles.length) {
                ret += resTitles[i].name + ': ' + String(rowData.results[i]) + '\n';
            } else {
                ret += `Metric ${i}: ` + String(rowData.results[i]) + '\n';
            }
        }
        
        return ret;
    }

    const contractKit = new ContractKit({        
        client: new APIClient({url: localStorage.getItem("bcEndpoint")}),        
    });

    useEffect (() => {
        loadUniqueTests();
    }, []);

    const [readingTests, setReadingTests] = useState(false);

    // NEW: Fetch test names from MongoDB (unique_test_names collection)
    const fetchNamesFromMongoDB = async (prefix = '') => {
        try {
            console.log("📋 Fetching test names from MongoDB...");
            let url = configuration.passer.PythonNames || 'http://195.230.127.227:8302/getnames';
            
            // Add prefix parameter if provided
            if (prefix) {
                url += `?prefix=${encodeURIComponent(prefix)}`;
                console.log(`Filtering tests starting with: ${prefix}`);
            }
            
            const response = await axios.get(url);
            
            if (response.data && Array.isArray(response.data)) {
                console.log(`✅ Retrieved ${response.data.length} test names from MongoDB`);
                return response.data;
            } else {
                console.error("Invalid response format from MongoDB");
                return [];
            }
        } catch (error) {
            console.error("❌ Error fetching names from MongoDB:", error.message);
            return [];
        }
    };

    // NEW: Fetch scores from MongoDB (test_scores collection)
    const fetchScoresFromMongoDB = async (testid = null, prefix = '') => {
        try {
            console.log("📊 Fetching scores from MongoDB...");
            let url = configuration.passer.PythonTestData || 'http://195.230.127.227:8302/gettestdata';
            
            // Add parameters
            const params = new URLSearchParams();
            if (testid) {
                params.append('testid', testid);
                console.log(`Fetching specific test: ${testid}`);
            } else if (prefix) {
                params.append('testid_prefix', prefix);
                console.log(`Filtering scores for tests starting with: ${prefix}`);
            }
            params.append('limit', '1000');
            params.append('sort', 'created_at');
            params.append('order', 'desc');
            
            const fullUrl = `${url}?${params.toString()}`;
            console.log(`Fetching from: ${fullUrl}`);
            
            const response = await axios.get(fullUrl);
            
            if (response.data.success) {
                console.log(`✅ Retrieved ${response.data.count} test records from MongoDB`);
                return response.data.data;
            } else {
                console.error("MongoDB query failed:", response.data);
                return [];
            }
        } catch (error) {
            console.error("❌ Error fetching scores from MongoDB:", error.message);
            return [];
        }
    };

    // Load unique test names from MongoDB
    const loadUniqueTests = async (prefix = filterPrefix) => {
        console.log("🔄 loadUniqueTests - start", prefix ? `with prefix: ${prefix}` : '');
        setReadingTests(true);

        try {
            const namesData = await fetchNamesFromMongoDB(prefix);
            
            if (namesData && Array.isArray(namesData)) {
                const jsonArray = namesData.map(item => ({ name: item.testid }));
                
                SetUniqueTests(jsonArray);
                console.log(`✅ Loaded ${jsonArray.length} tests${prefix ? ` starting with '${prefix}'` : ''}`);
            } else {
                console.log("⚠️ No test names received from MongoDB");
                SetUniqueTests([]);
            }
        } catch (error) {
            console.error("❌ Error in loadUniqueTests:", error);
            SetUniqueTests([]);
        } finally {
            setReadingTests(false);
            console.log("✓ loadUniqueTests - end");
        }        
    }

    const handleFilterApply = () => {
        console.log(`Applying filter: ${filterPrefix}`);
        loadUniqueTests(filterPrefix);
    };

    const handleFilterClear = () => {
        console.log("Clearing filter");
        setFilterPrefix('');
        loadUniqueTests('');
    };

    const handleFilterMATests = () => {
        console.log("Filtering MA tests");
        setFilterPrefix('ma');
        loadUniqueTests('ma');
    };

    // NEW: Load all MA test scores from MongoDB
    const loadMAScoresFromMongoDB = async () => {
        console.log("📊 Loading ALL MA test scores from MongoDB...");
        setReadingData(true);
        
        try {
            const maTests = await fetchScoresFromMongoDB(null, 'ma');
            
            if (!maTests || maTests.length === 0) {
                console.log("⚠️ No MA tests found in MongoDB");
                alert("No MultiAgent tests found in MongoDB");
                setReadingData(false);
                return;
            }
            
            console.log(`✅ Found ${maTests.length} MA test records in MongoDB`);
            
            // Transform MongoDB data to match blockchain format
            const transformedTests = maTests.map((test, index) => ({
                id: index + 1,
                created_at: test.created_at,
                userid: test.creator,
                testid: test.testid,
                description: test.description || 'MultiAgent test from MongoDB',
                results: test.results,
                checksum: test.checksum,
                reference: test.reference,
                candidate: test.candidate
            }));
            
            setTestResults(transformedTests);
            setDataSource('mongodb');
            setSelectedTest(''); // Clear selection since we're showing all
            
            // Calculate averages and standard deviations
            calculateStatistics(transformedTests);
            
            alert(`✅ Loaded ${maTests.length} MA test results from MongoDB`);
            
        } catch (error) {
            console.error("❌ Error loading MA scores:", error);
            alert(`Error: ${error.message}`);
        } finally {
            setReadingData(false);
        }
    };

    // Calculate statistics helper function
    const calculateStatistics = (results) => {
        if (!results || results.length === 0) return;
        
        let averages = [];
        let stddevs = [];
        
        results.forEach(rowData => {
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
        
        stddevs = stddevs.map(stddev => stddev.count <= 0 ? 0 : 
            Math.sqrt((stddev.sumOfSquares - (stddev.sum * stddev.sum) / stddev.count) / stddev.count)
        );
        setStdDev(stddevs);
        
        // Update chart data
        updateChartData(results, averages, stddevs, chartToShow);
    };

    // Update chart data helper function
    const updateChartData = (results, averages, stddevs, metricIndex = 0) => {
        if (!results || results.length === 0) return;
        
        const averageValue = (!averages[metricIndex] || averages[metricIndex].length === 0) ? 0 : averages[metricIndex];
        const sDev = stddevs.length === 0 ? 0 : stddevs[metricIndex];
        
        setData({
            labels: results.map((r, i) => {
                const d = new Date(r.created_at);    
                return `Test ${d.toLocaleDateString()}T${d.toLocaleTimeString()}`;
            }),
            datasets: [
              {
                label: resTitles[metricIndex].name,
                data: results.map(test => parseFloat(test.results[metricIndex])),
                backgroundColor: resTitles[metricIndex].color,
                borderColor: resTitles[metricIndex].color.replace('0.4)', '1)'),
                borderWidth: 1,
                fill: true,
              },
              {
                label: 'Average: ' + averageValue.toFixed(4),
                data: Array(results.length).fill(averageValue),
                type: 'line',
                borderColor: '#000000',
                borderWidth: 0.3,
                fill: false,
                pointRadius: 0,
              },
              {
                label: 'StdDev: ' + sDev.toFixed(4),
                data: Array(results.length).fill(sDev),
                type: 'line',
                borderColor: 'darkblue',
                backgroundColor: 'darkblue',
                borderWidth: 0.5,
                fill: false,
                pointRadius: 0,
              }
            ],
        });
    };

    const [data, setData] = useState({});
    const options = {
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 0.001,
                    callback: function(value) {
                        return parseFloat(value).toFixed(5);
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
        
        if (testResults.length > 0) {
            updateChartData(testResults, average, stdDev, index);
        }
    }

    const [average, setAverage] = useState([]);
    const [stdDev, setStdDev] = useState([]);
    const [readingData, setReadingData] = useState(false);

    // NEW: Smart onRowSelect - MongoDB for 'ma' tests, blockchain for others
    const onRowSelect = async (e) => {
        setReadingData(true);
        setSelectedTest(e.data.name);
        const testID = e.data.name;
        
        try {
            // Check if test name starts with 'ma' - use MongoDB
            if (testID.toLowerCase().startsWith('ma')) {
                console.log(`🔵 Loading test "${testID}" from MongoDB...`);
                const mongoTests = await fetchScoresFromMongoDB(testID);
                
                if (mongoTests && mongoTests.length > 0) {
                    // Transform MongoDB data
                    const transformedTests = mongoTests.map((test, index) => ({
                        id: index + 1,
                        created_at: test.created_at,
                        userid: test.creator,
                        testid: test.testid,
                        description: test.description || 'MultiAgent test',
                        results: test.results,
                        checksum: test.checksum,
                        reference: test.reference,
                        candidate: test.candidate
                    }));
                    
                    setTestResults(transformedTests);
                    setDataSource('mongodb');
                    calculateStatistics(transformedTests);
                    
                    console.log(`✅ Loaded ${transformedTests.length} records from MongoDB`);
                    console.log(`📊 Total metrics per record: ${transformedTests[0].results.length}`);
                    if (transformedTests[0].checksum) {
                        console.log(`🔐 Checksum: ${transformedTests[0].checksum}`);
                    }
                } else {
                    console.log("⚠️ No data found in MongoDB, trying blockchain...");
                    await loadFromBlockchain(testID);
                }
            } else {
                // Load from blockchain for non-MA tests
                console.log(`🟢 Loading test "${testID}" from blockchain...`);
                await loadFromBlockchain(testID);
            }
        } catch (error) {
            console.error("❌ Error loading test data:", error);
            alert(`Error loading test: ${error.message}`);
        } finally {
            setReadingData(false);
        }
    };

    // Blockchain loading function
    const loadFromBlockchain = async (testID) => {
        console.log(`🔗 Loading test "${testID}" from blockchain...`);
        try {
            const contract = contractKit.load("llmtest");
            const cursor = (await contract).table('testtable').query({testid: testID});         
            const rows = await cursor.all();
            
            let filteredResults = getFiles(rows).filter(result => String(result.testid) === testID);    
            
            if (filteredResults.length > 0) {
                setTestResults(getFiles(filteredResults));
                setDataSource('blockchain');
                calculateStatistics(filteredResults);
                console.log(`✅ Loaded ${filteredResults.length} records from blockchain`);
            } else {
                console.log("⚠️ No records found in blockchain");
                setTestResults([]);
                alert(`No records found for test: ${testID}`);
            }
        } catch (error) {
            console.error("❌ Error loading from blockchain:", error);
            setTestResults([]);
            alert(`Error loading from blockchain: ${error.message}`);
        }
    };

    const calculateElapsedTime = (currentDate, previousDate) => {
        if (!previousDate) return "";
        
        const current = new Date(currentDate);
        const previous = new Date(previousDate);
        
        const diffInMs = current - previous;
        const diffInSeconds = Math.floor(diffInMs / 1000);
        
        return diffInSeconds;
    };

    const exportToExcel = () => {
        const XLSX = require('xlsx');
        const myData = [];

        for (var i = 0; i < testResults.length; i++) {
            const dt = new Date(testResults[i].created_at);
            var tmpRes = {
                id: Number(testResults[i].id), 
                date: String(testResults[i].created_at),
                userID: String(testResults[i].userid || testResults[i].creator),
                testID: String(testResults[i].testid),
                Description: testResults[i].description,
                DataSource: dataSource,
            }
            
            // Add checksum if available (MongoDB data)
            if (testResults[i].checksum) {
                tmpRes.Checksum = testResults[i].checksum;
            }
            
            // Export all available metrics
            const totalMetrics = testResults[i].results.length;
            for (var j = 0; j < totalMetrics; j++) {
                const value = testResults[i].results[j];
                const numValue = Number(value);
                
                const metricName = j < resTitles.length ? resTitles[j].name : `Metric ${j}`;
                tmpRes[metricName] = !isNaN(numValue) ? numValue : value;
            }           
            
            if (i > 0) {
                tmpRes["Elapsed time"] = calculateElapsedTime(testResults[i].created_at, testResults[i-1].created_at);
            } else {
                tmpRes["Elapsed time"] = "";
            }
            
            myData.push(tmpRes);
        }

        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(myData);
        XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
        const filename = `test_results_${selectedTest || 'all'}_${dataSource}_${Date.now()}.xlsx`;
        XLSX.writeFile(wb, filename);
        console.log(`📥 Exported to: ${filename}`);
    }

    const [filters, setFilters] = useState({
        global: { value: null, matchMode: FilterMatchMode.CONTAINS },
        name: { value: null, matchMode: FilterMatchMode.STARTS_WITH },
    });

    const chartRef = useRef();

    const downloadChart = () => {
        console.log('chartRef', chartRef.current);
        if (chartRef.current) {
            const canvas = chartRef.current.getCanvas();
            const base64Image = canvas.toDataURL("image/png");
            const link = document.createElement('a');
            link.href = base64Image;
            link.download = `chart_${selectedTest || 'all'}_${dataSource}_${Date.now()}.png`;
            link.click();
        }
    };

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            <div style={{ width: '12%'}}>
                <h3>Unique tests</h3>
                
                {/* Filter Controls */}
                <div style={{ marginBottom: '10px', padding: '5px' }}>
                    <label style={{ fontSize: '12px', fontWeight: 'bold', display: 'block', marginBottom: '5px' }}>
                        🔍 Filter by prefix:
                    </label>
                    <div style={{ display: 'flex', gap: '3px', marginBottom: '5px' }}>
                        <InputText 
                            value={filterPrefix}
                            onChange={(e) => setFilterPrefix(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleFilterApply()}
                            placeholder="e.g., ma"
                            style={{ width: '100%', fontSize: '12px', padding: '4px' }}
                        />
                    </div>
                    <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                        <Button 
                            icon="pi pi-filter" 
                            onClick={handleFilterApply}
                            tooltip="Apply Filter"
                            tooltipOptions={{ position: 'top' }}
                            size="small"
                            style={{ fontSize: '11px', padding: '4px 8px' }}
                        />
                        <Button 
                            icon="pi pi-times" 
                            onClick={handleFilterClear}
                            className="p-button-outlined"
                            tooltip="Clear Filter"
                            tooltipOptions={{ position: 'top' }}
                            size="small"
                            style={{ fontSize: '11px', padding: '4px 8px' }}
                        />
                        <Button 
                            label="MA" 
                            onClick={handleFilterMATests}
                            className="p-button-success"
                            tooltip="Show MultiAgent Tests"
                            tooltipOptions={{ position: 'top' }}
                            size="small"
                            style={{ fontSize: '11px', padding: '4px 8px' }}
                        />
                    </div>
                    
                    {/* Load All MA Scores Button */}
                    <Button 
                        label="📊 Load All MA" 
                        onClick={loadMAScoresFromMongoDB}
                        className="p-button-info"
                        tooltip="Load all MA test scores from MongoDB"
                        tooltipOptions={{ position: 'top' }}
                        size="small"
                        style={{ 
                            fontSize: '11px', 
                            padding: '4px 8px', 
                            marginTop: '5px', 
                            width: '100%' 
                        }}
                    />
                    
                    {filterPrefix && (
                        <div style={{ 
                            fontSize: '10px', 
                            color: '#666', 
                            marginTop: '5px',
                            fontStyle: 'italic' 
                        }}>
                            Filtering: "{filterPrefix}*"
                        </div>
                    )}
                    
                    {/* Data Source Indicator */}
                    {testResults.length > 0 && (
                        <div style={{ 
                            fontSize: '10px', 
                            color: dataSource === 'mongodb' ? '#2196F3' : '#4CAF50',
                            marginTop: '5px',
                            fontWeight: 'bold',
                            padding: '3px',
                            backgroundColor: dataSource === 'mongodb' ? '#E3F2FD' : '#E8F5E9',
                            borderRadius: '3px',
                            textAlign: 'center'
                        }}>
                            📍 {dataSource === 'mongodb' ? '🔵 MongoDB' : '🟢 Blockchain'}
                        </div>
                    )}
                </div>

                {readingTests ? (
                    <div className="p-d-flex p-flex-column p-ai-center" style={{ textAlign: 'center', padding: '1rem' }}>
                        <ProgressSpinner style={{ width: '50px', height: '50px' }} />
                        <div style={{ marginTop: '0.5rem' }}>Loading tests...</div>
                    </div>
                ) : (
                    <>
                        <div style={{ fontSize: '11px', color: '#666', marginBottom: '5px', padding: '0 5px' }}>
                            Found: {uniqueTests.length} test{uniqueTests.length !== 1 ? 's' : ''}
                        </div>
                        <DataTable 
                            style={{width: '90%'}} 
                            value={uniqueTests}   
                            onRowSelect={onRowSelect} 
                            selectionMode="single" 
                            selection={selectedTest}
                            rowClassName={rowClass} 
                            filters={filters} 
                            filterDisplay="row"
                            size={"small"} 
                            showGridlines 
                            stripedRows
                            emptyMessage="No tests found"
                        >
                            <Column field="name" header="Name" filter filterPlaceholder="By name"></Column>
                        </DataTable>
                    </>
                )}
            </div>

            <div style={{ width: '88%'}}>
                <h3>Test results</h3>
                <Button label="Export to Excel" style={{marginBottom: '10px'}} onClick={exportToExcel}/>
                <Button style={{marginBottom: '10px', marginLeft: '10px'}} label="Download chart" onClick={downloadChart} />
                
                {/* Data source indicator */}
                {testResults.length > 0 && (
                    <span style={{ 
                        marginLeft: '10px', 
                        fontSize: '12px', 
                        color: dataSource === 'mongodb' ? '#2196F3' : '#4CAF50',
                        fontWeight: 'bold'
                    }}>
                        📊 Data from: {dataSource === 'mongodb' ? '🔵 MongoDB' : '🟢 Blockchain'} ({testResults.length} records)
                    </span>
                )}
                
                <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
                    <div style={{ width: '20%'}}>
                        {readingData ? (
                            <div className="p-d-flex p-flex-column p-ai-center" style={{ textAlign: 'center', padding: '1rem' }}>
                                <ProgressSpinner style={{ width: '50px', height: '50px' }} />
                                <div style={{ marginTop: '0.5rem' }}>
                                    {dataSource === 'mongodb' ? 'Loading from MongoDB...' : 'Loading from blockchain...'}
                                </div>
                            </div>
                        ) : (
                            <DataTable style={{width: '90%'}} value={resTitles}   
                                onRowSelect={onChartSelect} selectionMode="single" selection={chartToShow}
                                rowClassName={rowClass1}
                                size={"small"} showGridlines stripedRows
                                scrollable scrollHeight="600px">
                                <Column field="name" header="Name"></Column>
                            </DataTable>
                        )}
                    </div>
                    <div style={{ width: '100%'}}>
                        <div className="chart-container" style={{ overflowX: 'auto', width: '100%' }}>
                            <Chart ref={chartRef} type="bar" data={data} options={options} style={{ minWidth: '2000px' }} />
                        </div>
                    </div>
                </div>
                
                {/* Display averages for all metrics */}
                {average.length > 0 ? (
                    <div style={{ 
                        width: '95%', 
                        border: '1px solid black', 
                        padding: '10px', 
                        margin: '20px auto', 
                        lineHeight: '1.5'
                    }}>
                        <h3 style={{ textAlign: 'center', marginBottom: '15px' }}>
                            📊 Averages Summary 
                            <span style={{ 
                                fontSize: '14px', 
                                marginLeft: '10px',
                                color: dataSource === 'mongodb' ? '#2196F3' : '#4CAF50'
                            }}>
                                ({dataSource === 'mongodb' ? '🔵 MongoDB' : '🟢 Blockchain'})
                            </span>
                        </h3>
                        
                        {/* Base NLP Metrics */}
                        <div style={{ marginBottom: '20px' }}>
                            <h4 style={{ 
                                backgroundColor: '#e8f5e9', 
                                padding: '8px', 
                                borderRadius: '4px',
                                marginBottom: '10px'
                            }}>Base NLP Metrics (24)</h4>
                            <div style={{ 
                                display: 'grid', 
                                gridTemplateColumns: 'repeat(3, 1fr)', 
                                gap: '10px' 
                            }}>
                                {average.slice(0, 24).map((avg, i) => (
                                    <p key={i} style={{ 
                                        margin: '0.3em 0', 
                                        fontSize: '13px',
                                        padding: '5px',
                                        backgroundColor: i % 2 === 0 ? '#f5f5f5' : 'white'
                                    }}>
                                        <strong>{resTitles[i].name}:</strong> {avg.toFixed(4)}
                                    </p>
                                ))}
                            </div>
                        </div>
                        
                        {/* MultiAgent Performance Metrics - Only show if they exist */}
                        {average.length > 24 && (
                            <div>
                                <h4 style={{ 
                                    backgroundColor: '#e3f2fd', 
                                    padding: '8px', 
                                    borderRadius: '4px',
                                    marginBottom: '10px'
                                }}>🤖 MultiAgent Performance Metrics ({average.length - 24})</h4>
                                <div style={{ 
                                    display: 'grid', 
                                    gridTemplateColumns: 'repeat(3, 1fr)', 
                                    gap: '10px' 
                                }}>
                                    {average.slice(24).map((avg, i) => {
                                        const metricIndex = i + 24;
                                        const metricName = resTitles[metricIndex] ? resTitles[metricIndex].name : `Metric ${metricIndex}`;
                                        
                                        // Format based on metric type
                                        let formattedValue;
                                        if (metricName.includes('Time') || metricName.includes('ms')) {
                                            formattedValue = `${avg.toFixed(2)} ms`;
                                        } else if (metricName.includes('Tokens') || metricName.includes('Messages') || 
                                                   metricName.includes('Turns') || metricName.includes('Agents') || 
                                                   metricName.includes('Rounds') || metricName.includes('Docs')) {
                                            formattedValue = Math.round(avg);
                                        } else if (metricName.includes('Rate') || metricName.includes('%')) {
                                            formattedValue = `${(avg * 100).toFixed(2)}%`;
                                        } else if (metricName.includes('Strategy')) {
                                            const strategies = ['Unknown', 'Collaborative', 'Sequential', 'Competitive', 'Hierarchical'];
                                            formattedValue = strategies[Math.round(avg)] || 'Unknown';
                                        } else if (metricName.includes('RAG')) {
                                            formattedValue = Math.round(avg) === 1 ? 'Yes' : 'No';
                                        } else {
                                            formattedValue = avg.toFixed(4);
                                        }
                                        
                                        return (
                                            <p key={metricIndex} style={{ 
                                                margin: '0.3em 0', 
                                                fontSize: '13px',
                                                padding: '5px',
                                                backgroundColor: i % 2 === 0 ? '#f5f5f5' : 'white'
                                            }}>
                                                <strong>{metricName}:</strong> {formattedValue}
                                            </p>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                ) : null}
                
                <DataTable style={{width: '90%'}} value={testResults}   
                    selectionMode="single"
                    rowClassName={rowClass}
                    size={"small"} showGridlines stripedRows
                    scrollable scrollHeight="400px">

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