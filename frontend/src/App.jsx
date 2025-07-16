/* eslint-disable no-unused-vars */
import { useState } from "react";
import Container from "./components/container";
import Navbar from "./components/Navbar";
import './App.css'
import Section from "./components/Section";
import Footer from "./components/Footer";


function App() {
  

  return (
    <main className="bg-[#201317] text-white w-full">
      <div className="min-h-screen flex flex-col">
      <Navbar/>
      <Container/>
      <Section />
      <Footer/>  
      </div>
    </main>

  );
}

export default App;
