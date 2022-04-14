import "./App.css";
import * as React from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import axios from "axios";
import { CSpinner } from "@coreui/bootstrap-react";
import 'bootstrap/dist/css/bootstrap.min.css'

function App() {
  const canvas = React.createRef(null);
  const [predicted, setPredicted] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);
  const predict = async (data) => {
    let body = {
      img: data,
    };
    const response = await axios.post(`http://127.0.0.1:5000/api`, body, {
      headers: {
        "Content-Type": "application/json",
      },
    });
    setPredicted(response.data);
    setIsLoading(false);
    return response;
  };
  return (
    <div className="content">
      <h1 style={{ textAlign: "center", marginTop: "2em",color:"white" }}>
        HANDWRITTEN CHARACTER PREDICT (BETA VERSION)
      </h1>
      <p
        style={{
          textAlign: "center",
          width: "40%",
          margin: "10px auto 20px",
          color: "#a1a1a1",
        }}
      >
        As mentioned before this machine only predicts characters, therefore do
        not throw the words into the black box or pay the piper! &#129325;
      </p>
      <div className="App">
        <div className="sketch">
          <ReactSketchCanvas
            ref={canvas}
            strokeWidth={20}
            width="40%"
            height="100%"
            strokeColor="black"
            canvasColor="white"
            style={{ margin: "auto", cursor: "crosshair" }}
          />
        </div>
        <div className="button-div">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              setPredicted("");
              setIsLoading(true);
              canvas.current
                .exportImage("png")
                .then((data) => {
                  predict(data);
                })
                .catch((e) => {
                  console.log(e);
                });
              //canvas.current.clearCanvas();
            }}
          >
            Predict Handwritten
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setPredicted("");
              setIsLoading(false);
              canvas.current.clearCanvas();
            }}
          >
            Reset Scratch
          </button>

          <button
            type="button"
            onClick={() => {
              setPredicted("");
              setIsLoading(false);
              canvas.current.undo();
            }}
            className="btn btn-warning"
          >
            Undo
          </button>
        </div>
        {isLoading === false ? (
          <>
            <p
              style={{
                textAlign: "center",
                width: "40%",
                margin: "3em auto 20px",
                color: "#a1a1a1",
              }}
            >
              You've been written the characters of
            </p>
            <h3 style={{color:"white"}}>{predicted}</h3>
          </>
        ) : (
          <>
            <h4 style={{color:"white"}}>Progress...</h4>
            <CSpinner color="light" />
            <h6 style={{color:"white"}}>Sabar ya neuronnya banyak :(</h6>
          </>
        )}
      </div>

      <footer>
        <div className="footerpart">
          <div className="copywright">
            <h6
              className="bold"
              style={{ textAlign: "center", color: "white" }}
            >
              Copyrights {new Date().getFullYear()} Thariq Hadyan
            </h6>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
