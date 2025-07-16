import React from 'react'

const Button = (props) => {
    return (
        <button
            className="rounded-full p-1 px-2 ml-10 cursor-pointer grdnt focus:outline-none"
            onClick={props.onClick}
            style={{
                backgroundColor: props.color,
            }}
        >
            <h3 className='text-[#201317] font-bold'>{props.title}</h3>
        </button>
    );
}

export default Button
