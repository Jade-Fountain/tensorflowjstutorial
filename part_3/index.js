// index.js
import 'bootstrap/dist/css/bootstrap.css'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import $ from 'jquery'
require('babel-polyfill')

import {MnistData} from './mnist_data'
import * as util from './mnist_utils'
import {initCanvas} from './draw_utils'

let data = new MnistData()
$('#load-data-btn').click(async()=>{
	let msg = $('#loading-data');
	msg.text('Downloading MNIST data....');
	await data.load(40000,10000);

	msg.toggleClass('badge-warning badge-success');
	msg.text('MNIST data downloaded!');
	//what is this? : $('#load-btn').prop('disabled',true);

	const [x_test,y_test] = data.getTestData(8);
	const labels = Array.from(y_test.argMax(1).dataSync());
	util.showExample('mnist-preview', x_test, labels);
});

$('input[name=optmodel]:radio').click(function() {
    $('#model').text(util.getModel(this.value))
});

let model = tf.sequential()    
$('#init-btn').click(function() {
    var md = $.trim($('#model').val())
    eval(md)
    tfvis.show.modelSummary($('#summary')[0], model)
    $('#train-btn').prop('disabled', false)
    $('#predict-btn').prop('disabled', false)
    $('#eval-btn').prop('disabled', false)
    $('#show-example-btn').prop('disabled', false)
});

let round = (num) => parseFloat(num*100).toFixed(1);

$('#train-btn').click(async() => {        
    var msg = $('#training')
    msg.toggleClass('badge-warning badge-success')    
    msg.text('Training, please wait...')    
    
    var epoch = parseInt($('#epoch').val())
    var batch = parseInt($('#batch').val())
    
    const [x_train, y_train] = data.getTrainData()
    
    let nIter = 0
    const numIter = Math.ceil(x_train.shape[0] / batch) * epoch    
    $('#num-iter').text('Num Training Iteration: '+ numIter)
        
    const trainLogs = []
    const loss = $('#loss-graph')[0]
    const acc = $('#acc-graph')[0]

    const history = await model.fit(x_train, y_train, {
        epochs: epoch,
        batchSize: batch,
        shuffle: true,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                nIter++
                trainLogs.push(logs)
                tfvis.show.history(loss, trainLogs, ['loss'], { width: 300, height: 160 })
                tfvis.show.history(acc, trainLogs, ['acc'], { width: 300, height: 160 })
                $('#train-iter').text(`Training..( ${round(nIter / numIter)}% )`)
                $('#train-acc').text('Training Accuracy : '+ round(logs.acc) +'%')
            },
        }
    })
    
    $('#train-iter').toggleClass('badge-warning badge-success')    
    msg.toggleClass('badge-warning badge-success')    
    msg.text('Training Done')
    $('#save-btn').prop('disabled', false)        
});



