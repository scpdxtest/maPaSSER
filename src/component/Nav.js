import { Menubar } from 'primereact/menubar';
// import "primereact/resources/themes/vela-blue/theme.css"; // edit vela-blue to change theme
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import { useState, useEffect } from 'react';
import { blueFontSmall } from './mylib';     
import {checkBCEndpoint, checkIPFSEndpoint} from './BCEndpoints.js';
import aboutIcon from './about.png';
import './Nav.css';
import testingIcon from './testing_icon-icons.com_72182.png'
import loginIcon from './Login_37128.png'

const Navigation = () => {
    const [selectedModel, setSelectedModel] = useState(localStorage.getItem("selectedLLMModel") || 'Default -> mixtral');
    const [selectedOllama, setSelectedOllama] = useState(localStorage.getItem("selectedOllama") || 'http://');
    const [ChromaDBPath, setChromaDBPath] = useState(localStorage.getItem("selectedChromaDB") || 'http://');
    const [multiAgentAPI, setMultiAgentAPI] = useState(localStorage.getItem("selectedMultiAgent") || 'http://');

    useEffect(() => {
        checkBCEndpoint().then(async (res) => {
            localStorage.setItem("bcEndpoint", res);
        });
        checkIPFSEndpoint().then(async (res) => {
            localStorage.setItem("ipfsEndpoint_host", res.host);
            localStorage.setItem("ipfsEndpoint_port", res.port);
        });
    }, []);

    const navlist = [
      { label: 'About', icon: <img src={aboutIcon} alt="About" width="22" height="22" />, command: () => {
          window.location.href = './#/about';
        }
      },
      {
        label: 'Create Vectorstore',
        icon: <span style={{ color: 'red' }} className="pi pi-fw pi-plus"></span>,
        items: [
          { label: 'From Text', icon: <span style={{ color: 'red' }} className="pi pi-fw pi-file"></span>, command: () => { window.location.href = './#/dbfromtext'; } },
          { label: 'From PDF/DOCX', icon: <span style={{ color: 'red' }} className="pi pi-fw pi-file-pdf"></span>, command: () => { window.location.href = './#/dbfrompdf'; } },
          { label: 'From WEB', icon: <span style={{ color: 'red' }} className="pi pi-fw pi-globe"></span>, command: () => { window.location.href = './#/dbfromweb'; } }
        ]
      },
      {
        label: 'Chat',
        icon: <span style={{ color: 'green' }} className="pi pi-fw pi-comments"></span>,
        items: [
          { label: 'Chat with LLM', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-comments"></span>, command: () => { window.location.href = './#/gpt'; } },
          { label: 'RAG Q&A with LLM', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-question-circle"></span>, command: () => { window.location.href = './#/chatfromdb'; } },
          { label: 'Chat with LLM MultiAgent', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-users"></span>, command: () => { window.location.href = './#/chatfromdbmultiagent'; } },
          { label: 'Chat over Picture', icon: <span style={{ color: '#0ea5a4' }} className="pi pi-fw pi-camera"></span>, command: () => { window.location.href = './#/chatoverpicture'; } },
          { label: 'Text to Image with LLM', icon: <span style={{ color: '#0ea5a4' }} className="pi pi-fw pi-image"></span>, command: () => { window.location.href = './#/t2imgchat'; } },
          { label: 'Image Edit with LLM', icon: <span style={{ color: '#0ea5a4' }} className="pi pi-fw pi-image"></span>, command: () => { window.location.href = './#/t2imgedit'; } },
          { label: 'Diagram Creation with LLM', icon: <span style={{ color: '#0ea5a4' }} className="pi pi-fw pi-sitemap"></span>, command: () => { window.location.href = './#/diagramchat'; } }
        ]
      },
      {
        label: 'Tests',
        icon: <img src={testingIcon} alt="Tests" width="20" height="20" />,
        items: [
          { label: 'Q&A Dataset', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-database"></span>, command: () => { window.location.href = './#/qatest'; } },
          { label: 'RAG Q&A Score Test', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-chart-line"></span>, command: () => { window.location.href = './#/testRAGbat'; } },
          { label: 'Q&A Score Test with Finetune', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-sliders-h"></span>, command: () => { window.location.href = './#/finetuneTestbat'; } },
          { label: 'Show Test Results', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-table"></span>, command: () => { window.location.href = './#/showTestResults'; } },
          { label: 'Q&A Time LLM Tests', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-clock"></span>, command: () => { window.location.href = './#/testtimebat'; } },
          { label: 'Show Time Test Results', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-table"></span>, command: () => { window.location.href = './#/showTimeTestResults'; } },
          { label: 'MultiAgent Single Test', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-users"></span>, command: () => { window.location.href = './#/testmultiagentbat'; } },
          { label: 'MultiAgent Batch Test', icon: <span style={{ color: 'green' }} className="pi pi-fw pi-users"></span>, command: () => { window.location.href = './#/testmultiagentbatch'; } }
        ]
      },
      {
        label: 'OCR',
        icon: <span style={{ color: 'blue' }} className="pi pi-fw pi-file-o"></span>,
        command: () => { window.location.href = './#/ocr'; }
      },
      {
        label: 'Configuration',
        icon: <span style={{ color: 'blue' }} className="pi pi-fw pi-cog"></span>,
        items: [
          { label: 'Settings', icon: <span style={{ color: 'blue' }} className="pi pi-fw pi-cog"></span>, command: () => { window.location.href = './#/selectmodel'; } },
          { label: 'Add/Remove Model', icon: <span style={{ color: 'blue' }} className="pi pi-fw pi-plus-circle"></span>, command: () => { window.location.href = './#/addmodel'; } }
        ]
      },
      { label: 'ManageDB', icon: <span style={{ color: 'purple' }} className="pi pi-fw pi-server"></span>, command: () => { window.location.href = './#/managedb'; } },
      { label: 'AnchorLogin', icon: <span style={{ color: 'orange' }} className="pi pi-fw pi-key"></span>, command: () => { window.location.href = './#/testwharf'; } }
    ];
    
    return(

        <header>
            <nav>
                <ul>
                    <Menubar 
                        model={navlist} 
                        end={
                            <div>
                                <div style={blueFontSmall}><b>OllamaAPI:</b>{selectedOllama} | <b>Model:</b>{selectedModel}</div>
                                <div style={blueFontSmall}><b>ChromaAPI:</b>{ChromaDBPath} | <b>BCName:</b>{localStorage.getItem('wharf_user_name')}</div>
                                <div style={blueFontSmall}><b>MultiAgentAPI:</b>{multiAgentAPI}</div>
                            </div>
                        }
                    />
                </ul>
            </nav>
         </header>


        // <div style={{ display: 'flex', width: '100%'}}>
        //     <div style={{width: '10%', display: 'flex'}} >
        //         <TieredMenu model={navlist} />
        //         <span style={{marginLeft: '20px'}}>
        //             <div style={blueFontSmall}>
        //                 <br/>
        //                 {selectedOllama} | {selectedModel} | {ChromaDBPath}
        //             </div>
        //         </span>
        //     </div>
        // </div>
    )

}

export default Navigation;