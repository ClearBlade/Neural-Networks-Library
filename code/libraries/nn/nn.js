function NeuralNetwork(inputnodes, hiddennodes, outputnodes, learning_rate, activation) {

    if(inputnodes == undefined){
        inputnodes = 3;
    }

    if(hiddennodes == undefined){
        hiddennodes = 16;
    }

    if (arguments[0] instanceof NeuralNetwork) {
        var nn = arguments[0];
        this.inodes = nn.inodes;
        this.hnodes = nn.hnodes;
        this.onodes = nn.onodes;
        this.wih = nn.wih.copy();
        this.who = nn.who.copy();
        this.activation = nn.activation;
        this.derivative = nn.derivative;
        this.lr = this.lr;
    } else {
        this.inodes = inputnodes;
        this.hnodes = hiddennodes;
        this.onodes = outputnodes;

        this.wih = new Matrix(this.hnodes, this.inodes);
        this.who = new Matrix(this.onodes, this.hnodes);

        this.wih.randomize();
        this.who.randomize();

        this.lr = learning_rate || 0.1;

        if (activation == 'tanh') {
            this.activation = NeuralNetwork.tanh;
            this.derivative = NeuralNetwork.dtanh;
        } else {
            this.activation = NeuralNetwork.sigmoid;
            this.derivative = NeuralNetwork.dSigmoid;
        }
    }
}


NeuralNetwork.prototype.copy = function () {
    return new NeuralNetwork(this);
}

NeuralNetwork.prototype.mutate = function () {
    this.wih = Matrix.map(this.wih, mutate);
    this.who = Matrix.map(this.who, mutate);
}

NeuralNetwork.prototype.train = function (inputs_array, targets_array) {
    var inputs = Matrix.fromArray(inputs_array);
    var targets = Matrix.fromArray(targets_array);

    var hidden_inputs = Matrix.dot(this.wih, inputs);
    var hidden_outputs = Matrix.map(hidden_inputs, this.activation);
    var output_inputs = Matrix.dot(this.who, hidden_outputs);
    var outputs = Matrix.map(output_inputs, this.activation);
    var output_errors = Matrix.subtract(targets, outputs);

    var whoT = this.who.transpose();
    var hidden_errors = Matrix.dot(whoT, output_errors)

    var gradient_output = Matrix.map(outputs, this.derivative);
    gradient_output.multiply(output_errors);
    gradient_output.multiply(this.lr);
    var gradient_hidden = Matrix.map(hidden_outputs, this.derivative);
    gradient_hidden.multiply(hidden_errors);
    gradient_hidden.multiply(this.lr);
    var hidden_outputs_T = hidden_outputs.transpose();
    var deltaW_output = Matrix.dot(gradient_output, hidden_outputs_T);
    this.who.add(deltaW_output);

    var inputs_T = inputs.transpose();
    var deltaW_hidden = Matrix.dot(gradient_hidden, inputs_T);
    this.wih.add(deltaW_hidden);
}

NeuralNetwork.prototype.predict = function (inputs_array) {
    var inputs = Matrix.fromArray(inputs_array);
    var hidden_inputs = Matrix.dot(this.wih, inputs);
    var hidden_outputs = Matrix.map(hidden_inputs, this.activation);
    var output_inputs = Matrix.dot(this.who, hidden_outputs);
    var outputs = Matrix.map(output_inputs, this.activation);
    return outputs.toArray();
}

NeuralNetwork.prototype.get_error = function (train_x, train_y) {
    error = 0;
    for (i = 0; i < train_x.length; i++) {
        train = train_y[i];
        pred = this.predict(train_x[i]);

        // Mean Absolute Error (MAE)
        var c = train.map(function (v, i) { return Math.abs(v - pred[i]); });
        error += c.reduce(function (a, b) { return a + b; }, 0);
    }
    return error / train_x.length;
}

NeuralNetwork.prototype.fit = function(train_x, train_y, test_x, test_y, epochs) {

    if(epochs == undefined){
        epochs = 4;
    }
    
    run_log = "Training on " + train_x.length + ". Testing on " + test_x.length + "\r";
    
    for (var i = train_x.length, train_idx = []; i--;) train_idx.push(i);
    
    for (e = 0; e < epochs; e++) {
        train_idx = shuffle(train_idx);
        
        start = Date.now();
        for (i = 0; i < train_idx.length; i++) {
            k = train_idx[i];
            this.train(train_x[k], train_y[k]);
        }

        epoch_time = Date.now() - start;
        loss = this.get_error(train_x, train_y);
        val_loss = this.get_error(test_x, test_y);
        
        run_log += "EPOCH " + (e + 1) + " - ";
        run_log += "time: " + (epoch_time / 1000).toFixed(2) + "s - ";
        run_log += "loss: " + loss.toFixed(2) + " ";
        run_log += "val_loss: " + val_loss.toFixed(2) + "\r";
    }
    
    return run_log
}

// Matrix Helper Functions
function Matrix(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.matrix = new Array(rows);
    for (var i = 0; i < this.rows; i++) {
        this.matrix[i] = new Array(cols);
        for (var j = 0; j < this.cols; j++) {
            this.matrix[i][j] = 0;
        }
    }
}

Matrix.prototype.randomize = function () {
    for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
            this.matrix[i][j] = randomGaussian();
        }
    }
}

Matrix.prototype.toArray = function () {
    var arr = [];
    for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
            arr.push(this.matrix[i][j]);
        }
    }
    return arr;
}

Matrix.prototype.transpose = function () {
    var result = new Matrix(this.cols, this.rows);
    for (var i = 0; i < result.rows; i++) {
        for (var j = 0; j < result.cols; j++) {
            result.matrix[i][j] = this.matrix[j][i];
        }
    }
    return result;
}

Matrix.prototype.copy = function () {
    var result = new Matrix(this.rows, this.cols);
    for (var i = 0; i < result.rows; i++) {
        for (var j = 0; j < result.cols; j++) {
            result.matrix[i][j] = this.matrix[i][j];
        }
    }
    return result;
}

Matrix.prototype.add = function (other) {
    if (other instanceof Matrix) {
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                this.matrix[i][j] += other.matrix[i][j];
            }
        }
    } else {
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                this.matrix[i][j] += other;
            }
        }
    }
}

Matrix.prototype.multiply = function (other) {
    if (other instanceof Matrix) {
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                this.matrix[i][j] *= other.matrix[i][j];
            }
        }
    } else {
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                this.matrix[i][j] *= other;
            }
        }
    }
}

Matrix.map = function (m, fn) {
    var result = new Matrix(m.rows, m.cols);
    for (var i = 0; i < result.rows; i++) {
        for (var j = 0; j < result.cols; j++) {
            result.matrix[i][j] = fn(m.matrix[i][j]);
        }
    }
    return result;
}

Matrix.subtract = function (a, b) {
    var result = new Matrix(a.rows, a.cols);
    for (var i = 0; i < result.rows; i++) {
        for (var j = 0; j < result.cols; j++) {
            result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j];
        }
    }
    return result;
}

Matrix.dot = function (a, b) {
    if (a.cols != b.rows) {
        console.log("Incompatible matrix sizes!");
        return;
    }
    var result = new Matrix(a.rows, b.cols);
    for (var i = 0; i < a.rows; i++) {
        for (var j = 0; j < b.cols; j++) {
            var sum = 0;
            for (var k = 0; k < a.cols; k++) {
                sum += a.matrix[i][k] * b.matrix[k][j];
            }
            result.matrix[i][j] = sum;
        }
    }
    return result;
}

Matrix.fromArray = function (array) {
    var m = new Matrix(array.length, 1);
    for (var i = 0; i < array.length; i++) {
        m.matrix[i][0] = array[i];
    }
    return m;
}

function randomGaussian() {
    return ((Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random()) - 3) / 3;
}

// Activation Functions
NeuralNetwork.sigmoid = function (x) {
    var y = 1 / (1 + Math.pow(Math.E, -x));
    return y;
}

NeuralNetwork.dSigmoid = function (x) {
    return x * (1 - x);
}

NeuralNetwork.tanh = function (x) {
    var y = Math.tanh(x);
    return y;
}

NeuralNetwork.dtanh = function (x) {
    var y = 1 / (Math.pow(Math.cosh(x), 2));
    return y;
}

function mutate(x) {
    if (random(1) < 0.1) {
        var offset = randomGaussian() * 0.5;
        var newx = x + offset;
        return newx;
    } else {
        return x;
    }
}

function validationSplit(data_x, data_y, validation_split) {

    if(validation_split == undefined){
        validation_split = 0.05;
    }

    // Split Training and Testing Data
    for (var i = data_x.length, data_idx = []; i--;) data_idx.push(i);
    data_idx = shuffle(data_idx);
    
    var train_x = [];
    var train_y = [];
    var test_x = [];
    var test_y = [];
    
    for (i = 0; i < data_idx.length; i++) {
        k=data_idx[i]
        if (i > data_idx.length * validation_split) {
            train_x.push(data_x[k]);
            train_y.push(data_y[k]);
        } else {
            test_x.push(data_x[k]);
            test_y.push(data_y[k]);
        }
    }    
     return [train_x, train_y, test_x, test_y]
}

function onehotEncode(y_series){
    keys = y_series.filter(onlyUnique);
    outputs = keys.length;
    
    var data_y = [];
    for (i = 0; i < y_series.length; i++) {    
        onehot = Array.apply(null, Array(outputs)).map(Number.prototype.valueOf,0);
        onehot[keys.indexOf(y_series[i])] = 1;
        data_y.push(onehot);
    }        
    
    return [data_y, outputs]
}

function onlyUnique(value, index, self) { 
    return self.indexOf(value) === index;
}



function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;
    while (0 !== currentIndex) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }
    return array;
}

// Data Gathering Function

function getData(req, modelName, inputs) {

    ClearBlade.init({ request: req });
    var query = ClearBlade.Query({ collectionName: modelName });
    query.fetch(function (err, d) { data = d.DATA });

    if(inputs == undefined){
        inputs = 3;
    }

    function chunk_overlap(array, chunkSize) {
        var retArr = [];
        for (var i = 0; i < array.length - (chunkSize - 1); i++) {
            retArr.push(array.slice(i, i + chunkSize));
        }
        return retArr;
    }
    var data_x = [];
    var y_series = [];
    
     for (i = 0; i < data.length; i++) { 
        x_series = chunk_overlap(data[i]["power_temperature_accelerometer"].split(',').map(parseFloat), inputs);   //power_temperature_accelerometer
        data_x.push(x_series);
        
        for (j = 0; j < x_series.length; j++) {         
            y_series.push(data[i]['maintenance_required']); //maintenance_required
        }
    }
    y_encoded = onehotEncode(y_series)

    data_x = [].concat.apply([], data_x);
    data_y = y_encoded[0]
    outputs = y_encoded[1]

    return [data_x, data_y, outputs];
}

// Saving and Loading Model Functions


var saveModel = function(resp, dataName, nn) {
    var collection = ClearBlade.Collection({collectionName:"nn_models"});

    var addRow = {
        dataset: dataName,
        datetime: Date.now().toString(),
        who: nn.who.matrix.toString(),
        wih: nn.wih.matrix.toString(),
    };
     var callback = function(err, data){
        if (err) {
            resp.error(data);
        }
        else {
            //resp.success(JSON.stringify(addRow));
        }
    };
    collection.create(addRow, callback);
};

var loadModel = function(resp, dataName, nn){
    
    var q = ClearBlade.Query({collectionName:"nn_models"} );
    q.equalTo("dataset", dataName);
    q.descending("datetime")
    

    function split_array(a, size)
    {
        var arrays = []
        while (a.length > 0)
        arrays.push(a.splice(0, size));    
        return arrays
    }
    
     var callback = function(err, data){
        if (err) {
            resp.error(data);
        } else {
            who_matrix = split_array(data.DATA[0].who.split(",").map(parseFloat),nn.who.cols);
            wih_matrix = split_array(data.DATA[0].wih.split(",").map(parseFloat),nn.wih.cols);
        }
     }
        
    
    q.fetch(callback)
    nn.who.matrix = who_matrix
    nn.wih.matrix = wih_matrix
}

