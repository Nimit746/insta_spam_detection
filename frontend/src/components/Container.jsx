/* eslint-disable no-unused-vars */
import React, { useState } from 'react'
import axios from "axios";
import Button from './Button';
import { Search } from 'lucide-react';



const Container = () => {
    const [text, setText] = useState("")
    const [result, setResult] = useState("")



    const handlePredict = async () => {
        try {
            const res = await axios.post("http://localhost:8000/predict", { text });
            const label = res.data.prediction;
            const prob = (res.data.probability * 100).toFixed(2);
            setResult(`Prediction: ${label}(${prob} %)`);
        } catch (error) {
            setResult("Error connecting to server.");
            console.error(error);
        }
    };
    return (

        <main>

            <div
                className="p-4 w-full max-w-[60%] mx-auto flex items-center justify-center flex-col mt-20 gradient rounded-2xl bg-no-repeat bg-cover bg-center h-[60vh]"
            // style={{
            //     backgroundImage: 'url("https://img.freepik.com/premium-vector/spam-detection-concept-ai-filte…lutter-ensuring-email-integrity-user-protection-efficient_277904-30558.jpg ")'
            // }}
            >
                <h1 className='font-extrabold text-5xl text-black self-center'>Detect Spam on Social Media</h1>
                <p className='text-black font-bold text-xl text-center mt-4'>Enter a post URL or username to analyze for spam...</p>

                <div className="flex self-center mt-10 bg-[#201317] py-2 p-2 rounded-xl items-center">
                    <Search size={23} />
                    <input
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Enter something..."
                        className="border-none p-2 h-7 self-center outline-none"
                    />
                    <Button
                        onClick={handlePredict}
                        className="ml-2 p-2 bg-blue-500 text-white"
                        title='Analyze'
                    />
                </div>

            </div>
            <div className="mt-4 text-center text-white text-xl font-semibold">{result}</div>



        </main>
    );
}

export default Container