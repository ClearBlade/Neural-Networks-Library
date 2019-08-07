function nn_test(req, resp) {

    // --------> Provide Data Collection Name <-------- //
        
        var dataName = 'Machine1'

    // --------> Hyperparameters <-------- //
    
        // inputs : (default = 3), hidden : (default = 16), validation_split : (default = 0.05), epochs : (default = 4)
        var inputs = 3
        var batch = 100;

    // --------> Data Preparation <-------- //

        var datasets = getData(req, dataName, inputs);          // Parameters Required : req, collection_name; Optional : inputs (default = 3)
        var data_x = datasets[0]                                   // Features
        var data_y = datasets[1]                                   // One Hot encoded Labels
        var outputs = datasets[2]                                  // Number of output Labels
        
        data_x = data_x.slice(data_x.length - batch)
        data_y = data_y.slice(data_x.length - batch)
        
        datasets = validationSplit(data_x, data_y, undefined)    // Parameters Required : Features, One Hot encoded Labels; Optional : validation_split measure (default = 0.05)
        var train_x = datasets[0]                                // Training Data  
        var train_y = datasets[1]                                // Training Labels
        var test_x = datasets[2]                                 // Validation Data
        var test_y = datasets[3]                                 // Validation Labels
    
    // --------> Define Neural Network Model <-------- //
    
        var nn = new NeuralNetwork(inputs, undefined, outputs);        // Optional Parameters : inputs (default = 3), hidden (default = 16)

    // --------> Run Neural Network Model <-------- //
    
        run_log = nn.fit(train_x, train_y, test_x, test_y, undefined)      // Optional Parameter : epochs (default = 4)
        log(run_log);
    
    // --------> Save the Model in a Collection <-------- //

        // var saveCollectionName = "nn_models";
        // saveModel(resp, saveCollectionName, nn);

    // --------> Load the Model from a Collection <-------- //

        // var loadCollectionName = "nn_models";
        // loadModel(resp, loadCollectionName, nn)

    // --------> Predict Using the model <-------- //
  
        var prediction = nn.predict([1900, 90, 1.7000])
        resp.success(prediction);
}
