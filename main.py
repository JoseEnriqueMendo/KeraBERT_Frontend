from flask import Flask, render_template,  request, redirect, url_for, flash
from flaskext.mysql import MySQL
import pymysql
import casoClinico
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import random
import time
from textwrap import wrap


app = Flask(__name__)

app.secret_key = "Tesis-BERT"

mysql = MySQL()

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'tesiskerabert'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_PORT'] = 3306

mysql.init_app(app)

parrafo=""

@app.route("/")
def inicio():
    data = {
        'titulo': 'Informacion',
    }
    return render_template('information.html', data=data)


@app.route("/pacientes")
def pacientes():
    title= "Pacientes"
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    
    cur.execute('SELECT * FROM Pacientes')
    data = cur.fetchall()
    cur.close()
    
    return render_template('pacientes.html', title=title, pacientes = data)

@app.route("/consultas")
def consultas():
    title= "consultas"
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    
    cur.execute('SELECT * FROM consulta')
    data = cur.fetchall()
    cur.close()
    return render_template('consultas.html', title=title, consulta= data)

@app.route("/creditos")
def creditos():
    data = {
        'titulo': 'Creditos',
    }
    return render_template('creditos.html', data=data)

@app.route("/nueva_consulta")
def nueva_consulta():
    title= "Registro de Consulta"
    return render_template('registro_consulta.html', title=title)

@app.route('/registro')
def registro():
    title= "Registro de Pacientes"
    return render_template('registro_paciente.html', title=title)

@app.route('/nuevo_paciente', methods=['POST'])
def nuevo_paciente():
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    if request.method == 'POST':
        dni = request.form['dni']
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        fecha_nacimiento = request.form['fechaNacimiento']
        genero = request.form['genero']
        telefono = request.form['telefono']
        
        cur.execute("INSERT INTO pacientes (Dni, Nombre, Apellido, Fecha_Nacimiento, Genero, Telefono) VALUES (%s,%s,%s,%s,%s,%s)", (dni, nombre, apellido, fecha_nacimiento, genero, telefono))
        cur.execute("INSERT INTO historialmedico (dni_paciente) VALUES (%s)", (dni))
        conn.commit()
        flash('Paciente agregado correctamente')
        return redirect(url_for('pacientes'))

@app.route('/editar/<dni>', methods=['POST', 'GET'])
def get_paciente(dni):
    title= "Editar Paciente"
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM pacientes WHERE Dni= %s", (dni))
    data = cur.fetchall()
    cur.close()
    print(data[0])
    return render_template('editar_paciente.html', paciente = data[0], title= title)

@app.route('/update/<dni>', methods=['POST'])
def update_employee(dni):
    if request.method == 'POST':
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        fecha_nacimiento = request.form['fechaNacimiento']
        genero = request.form['genero']
        telefono = request.form['telefono']
        conn = mysql.connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            UPDATE pacientes
            SET Nombre = %s,
                Apellido = %s,
                Fecha_Nacimiento = %s,
                Genero = %s,
                Telefono = %s
            WHERE Dni = %s
        """, (nombre, apellido, fecha_nacimiento, genero , telefono, dni))
        flash('Paciente actualizado correctamente')
        conn.commit()
        return redirect(url_for('pacientes'))

@app.route('/eliminar/<string:dni>', methods = ['POST','GET'])
def delete_employee(dni):
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute('DELETE FROM pacientes WHERE Dni = {0}'.format(dni))
    conn.commit()
    flash('Paciente eliminado correctamente')
    return redirect(url_for('pacientes'))


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

##########################################
set_seed(42)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    

max_len       = 302          # limitaciones porque google collab tiene un ram limitado
batch_size    = 16           # Paquetes de 16 elementos
nclases       = 3            # Comentarios positivos y negativos
num_epochs    = 2
learning_rate = 5e-5
    

class BertClassifier(nn.Module):

    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, nclases

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
        max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
        information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
        num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


model = BertClassifier()
model.bert.load_state_dict(torch.load('./modelbert2.pth'))
model.to(device)
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    #print("probs", probs)
    #print()
    return probs

def preprocessing_for_bert(data, tokenizer, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
    tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,             # Max length to truncate/pad
            padding="max_length",           # Pad sentence to max length, # pad_to_max_length=True
            truncation=True,
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def classifier_treatment(text):
    labels = ["ICRS", "CXL", "Keratoplasty"]
    text_pd = pd.Series(text)
    test_inputs, test_masks = preprocessing_for_bert(text_pd, tokenizer, max_len)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    probs = bert_predict(model, test_dataloader)
    preds = probs[:, 1]

    # Get accuracy over the test set
    id_y_pred = probs.argmax(axis=1)
    
    tratamiento = f"{labels[id_y_pred[0]]}"
    precision = f"{probs[0][id_y_pred[0]]:.3f}"
    
    resultados = [tratamiento, precision]
    return resultados
######################################################

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No se encontró ningún archivo en la solicitud"

        file = request.files['file']
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(file, dtype={'DNI': str})
            
            p = casoClinico.ClinicalCasesToParagraph(df)
            global parrafo 
            
            parrafo= p.returnParagraph()
            
            dni_paciente= df.iloc[0, 0]
            
            
            transposed_df = df.transpose()  # Transponer el DataFrame
                        
            table_html = transposed_df.to_html(index=True)
            # Redirigir a la misma página con los datos transpuestos
            return render_template('registro_consulta.html', table_html=table_html, dni_paciente= dni_paciente)
    
@app.route('/generar_consulta/<dni>', methods=['POST', 'GET'])
def generar_consulta(dni):
    resultados = classifier_treatment(parrafo)
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM Pacientes WHERE Dni= %s", (dni))
    data = cur.fetchall()
    cur.close()
    return render_template('registro_consulta.html', resultados= resultados, paciente = data[0], parrafo = parrafo)

@app.route('/registrar_caso_clinico/<dni>', methods=['POST'])
def registrar_caso_clinico(dni):
    conn = mysql.connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM historialmedico WHERE dni_paciente= %s", (dni))
    aux = cur.fetchall()
    historial = aux[0]
    print(historial)
    if request.method == 'POST':
        historialmedico = historial["id"]
        tratamiento = request.form['tratamiento']
        caso = request.form['parrafo']
        cur.execute("INSERT INTO consulta (historial_medico, tratamiento, parrafo) VALUES (%s,%s,%s)", (historialmedico,tratamiento,caso))
        conn.commit()
        flash('Registro agregado correctamente')
        return redirect(url_for('consultas'))        

if __name__=='__main__':
    app.run(debug=True)
