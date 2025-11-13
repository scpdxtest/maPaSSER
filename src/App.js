// import logo from './logo.svg';
import './App.css';
import "primereact/resources/themes/lara-light-teal/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import { Routes, Route } from 'react-router-dom';
import Navigation from './component/Nav';
import About from './component/About';
import ErrorBoundary from './component/ErrorBoundry';
import Gpt from './component/gpt';
import SelectModel from './component/SelectModel';
import DBFromText from './component/dbFromText';
import DBFromWEB from './component/dbFromWEB';
import ChatFromDB from './component/chatFromDB';
import DBFromPDF from './component/dbFromPDF';
import AddModel from './component/addModel';
import TestWharf from './component/TestWharf';
import TestRAGbat from './component/testRAGbat';
import ShowTestResults from './component/showTestResults';
import QATest from './component/qaTest';
import RagQATest from './component/ragQUTest';
import TestTimebat from './component/testTimebat';
import ShowTimeTestResults from './component/showTimeTestResults';
import ManageDB from './component/manageDB';
import ChatFromDBMultiAgent from './component/chatFromDB_multy_agent';
import FinetuneTestbat from './component/finetuneTestbat';
import ChatOverPicture from './component/chatOverPicture';
import OCR from './component/ocr';
import T2ImgChat from './component/t2imgCHAT';
import T2ImgEdit from './component/ImageEditChat';
import DiagramChat from './component/diagramChat';
import TestMultiAgentBat from './component/testMultiAgentBat';
import TestMultiAgentBatch from './component/testMultyAgentPipeLine';

// stable chromadb lib version: 1.10.4 !!!!!

function App() {
  return (
    <div className="App">
      <ErrorBoundary>   
       <Navigation />
            <Routes>
              <Route path='/about' element={<About/>}/>
              <Route path="/gpt" element={<Gpt/>} />
              <Route path="/selectmodel" element={<SelectModel/>} />
              <Route path="/chatfromdb" element={<ChatFromDB/>} />
              <Route path="/dbfromtext" element={<DBFromText/>} />
              <Route path="/dbfromweb" element={<DBFromWEB/>} />
              <Route path="/dbfrompdf" element={<DBFromPDF/>} />
              <Route path="/addmodel" element={<AddModel/>} />
              <Route path="/testwharf" element={<TestWharf/>} />
              <Route path="/testRAGbat" element={<TestRAGbat/>} />
              <Route path="/showTestResults" element={<ShowTestResults/>} />
              <Route path="/qatest" element={<QATest/>} />
              <Route path="/ragqatest" element={<RagQATest/>} />
              <Route path="/testtimebat" element={<TestTimebat/>} />
              <Route path="/showTimeTestResults" element={<ShowTimeTestResults/>} />
              <Route path="/managedb" element={<ManageDB/>} />
              <Route path="/chatfromdbmultiagent" element={<ChatFromDBMultiAgent/>} />
              <Route path="/finetuneTestbat" element={<FinetuneTestbat/>} />
              <Route path="/chatoverpicture" element={<ChatOverPicture/>} />
              <Route path="/ocr" element={<OCR/>} />
              <Route path="/t2imgchat" element={<T2ImgChat/>} />
              <Route path="/t2imgedit" element={<T2ImgEdit/>} />
              <Route path="/diagramchat" element={<DiagramChat/>} />
              <Route path="/testmultiagentbat" element={<TestMultiAgentBat/>} />
              <Route path="/testmultiagentbatch" element={<TestMultiAgentBatch/>} />
            </Routes>
        </ErrorBoundary>   
    </div>
  );
}

export default App;
