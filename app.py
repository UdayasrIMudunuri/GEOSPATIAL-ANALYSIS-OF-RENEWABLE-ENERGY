from flask import Flask,render_template,request
from state import stateImp
from generateDataset import generate
from biomass import BiomassPredict
from wind import windPredict
from solar import solarPredict
from smallhydro import smallHydroPredict
app=Flask(__name__)

state=""
ptype=""
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    return render_template('Predictions.html')

@app.route('/Regions')
def Regions():
    return render_template('Regions.html')

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/algorithms")
def algorithms():
    return render_template("algorithms.html")

@app.route('/north')
def north():
    return render_template('north.html')

@app.route('/Chandigarh')
def Chandigarh():
    fres=stateImp('Chandigarh')
    return render_template("northres.html",fres=fres)

@app.route('/Delhi')
def Delhi():
    fres=stateImp('delhi')
    return render_template("northres.html",fres=fres)   

@app.route('/Haryana')
def Haryana():
    fres=stateImp('Haryana')
    return render_template("northres.html",fres=fres)   

@app.route('/HimachalPradesh')
def HimachalPradesh():
    fres=stateImp('hp')
    return render_template("northres.html",fres=fres)

@app.route('/JammuKashmir')
def JammuKashmir():
    fres=stateImp('jammu')
    return render_template("northres.html",fres=fres)

@app.route('/Ladakh')
def Ladakh():
    fres=stateImp('Ladakh')
    return render_template("northres.html",fres=fres)

@app.route('/Punjab')
def Punjab():
    fres=stateImp('Punjab')
    return render_template("northres.html",fres=fres)

@app.route('/Rajasthan')
def Rajasthan():
    fres=stateImp('Rajasthan')
    return render_template("northres.html",fres=fres)

@app.route('/UttarPradesh')
def UttarPradesh():
    fres=stateImp('uttar_pradesh')
    return render_template("northres.html",fres=fres)

@app.route('/Uttarakhand')
def Uttarakhand():
    fres=stateImp('Uttarakhand')
    return render_template("northres.html",fres=fres)

#Eastern Region

@app.route('/Bihar')
def Bihar():
    fres=stateImp('Bihar')
    return render_template("eastres.html",fres=fres)

@app.route('/Jharkhand')
def Jharkhand():
    fres=stateImp('Jharkhand')
    return render_template("eastres.html",fres=fres)

@app.route('/Odisha')
def Odisha():
    fres=stateImp('orissa')
    return render_template("eastres.html",fres=fres) 

@app.route('/Sikkim')
def Sikkim():
    fres=stateImp('Sikkim')
    return render_template("eastres.html",fres=fres) 


@app.route('/WestBengal')
def WestBengal():
    fres=stateImp('Westbengal')
    return render_template("eastres.html",fres=fres) 

@app.route('/ArunachalPradesh')
def Arunachalpradesh():
    fres=stateImp('Arunachal_Pradesh')
    return render_template("eastres.html",fres=fres)

@app.route('/Assam')
def Assam():
    fres=stateImp('Assam')
    return render_template("eastres.html",fres=fres)

@app.route('/Manipur')
def Manipur():
    fres=stateImp('Manipur')
    return render_template("eastres.html",fres=fres)

@app.route('/Meghalaya')
def Meghalaya():
    fres=stateImp('Meghalaya')
    return render_template("eastres.html",fres=fres)

@app.route('/Mizoram')
def Mizoram():
    fres=stateImp('Mizoram')
    return render_template("eastres.html",fres=fres)

@app.route('/Nagaland')
def Nagaland():
    fres=stateImp('Nagaland')
    return render_template("eastres.html",fres=fres)

@app.route('/Tripura')
def Tripura():
    fres=stateImp('Tripura')
    return render_template("eastres.html",fres=fres)

#west Region

@app.route('/Chhattisgarh')
def Chhattisgarh():
    fres=stateImp('chattisgarh')
    return render_template("westres.html",fres=fres)

@app.route('/Gujarat')
def Gujarat():
    fres=stateImp('gujarat')
    return render_template("westres.html",fres=fres)

@app.route('/MadhyaPradesh')
def MadhyaPradesh():
    fres=stateImp('mp')
    return render_template("westres.html",fres=fres)


@app.route('/Maharashtra')
def Maharastra():
    fres=stateImp('maharashtra')
    return render_template("westres.html",fres=fres)


@app.route('/Goa')
def Goa():
    fres=stateImp('goa')
    return render_template("westres.html",fres=fres)

@app.route('/Hydro')
def HydroBoosting():
    global ptype
    print(state)
    rmse_ensemble,mape_ensemble,r2_ensemble=SmallHydro(state)

    return render_template('north.html',rmse_ensemble=rmse_ensemble,mape_ensemble=mape_ensemble,r2_ensemble=r2_ensemble)

@app.route('/Wind')
def WindBoosting():
    global ptype
    print(state)
    rmse_ensemble,mape_ensemble,r2_ensemble=Wind(state)

    return render_template('north.html',rmse_ensemble=rmse_ensemble,mape_ensemble=mape_ensemble,r2_ensemble=r2_ensemble)
@app.route('/Solar')
def SolarBoosting():
    global ptype
    print(state)
    rmse_ensemble,mape_ensemble,r2_ensemble=Solar(state)

    return render_template('north.html',rmse_ensemble=rmse_ensemble,mape_ensemble=mape_ensemble,r2_ensemble=r2_ensemble)

@app.route('/Biomass')
def BiomassBoosting():
    global ptype
    print(state)
    rmse_ensemble,mape_ensemble,r2_ensemble=BIOMASS(state)

    return render_template('north.html',rmse_ensemble=rmse_ensemble,mape_ensemble=mape_ensemble,r2_ensemble=r2_ensemble)

@app.route('/south')
def south():
    return render_template('south.html')

@app.route('/east')
def east():
    return render_template('east.html')

@app.route('/west')
def west():
    return render_template('west.html')

@app.route('/AndhraPradesh')
def AndhraPradesh():
    fres=stateImp('andhra_pradesh')
    return render_template("southres.html",fres=fres)   

@app.route('/Telangana')
def Telangana():
    fres=stateImp('Telangana')
    return render_template("southres.html",fres=fres)   

@app.route('/Karnataka')
def Karnataka():
    fres=stateImp('Karnataka')
    return render_template("southres.html",fres=fres)   

@app.route('/Kerala')
def Kerala():
    fres=stateImp('Kerala')
    return render_template("southres.html",fres=fres)   

@app.route('/TamilNadu')
def TamilNadu():
    fres=stateImp('Tamil_Nadu')
    return render_template("southres.html",fres=fres)   

@app.route('/Predictions')
def Predictions():
    return render_template('Predictions.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/Energy')
def Energy():
    return render_template('Energy.html')

@app.route('/predictpower',methods=['GET'])
def predictpower():
    state=request.args['state']
    power=request.args['power']
    #state = "data/" + state + ".xlsx"
    
    power=power.lower()
    sname=""
    print("State is "+state)
    if state=='AP':
        sname='andhra_pradesh'
    elif state=='AR':
        sname='Arunachal_Pradesh'
    elif state=='AS':
        sname='Assam'
    elif state=='BR':
        sname='Bihar'
    elif state=='CT':
        sname='Chandigarh'
    elif state=='GA':
        sname='gujarat'
    elif state=='HR':
        sname='haryana'
    elif state=='HP':
        sname='hp'
    elif state=='JK':
        sname='jammu'
    elif state=='GA':
        sname='goa'
    elif state=='JH':
        sname='Jharkhand'
    elif state=='KA':
        sname='karnataka'
    elif state=='KL':
        sname='kerala'
    elif state=='MP':
        sname='mp'
    elif state=='MH':
        sname='maharashtra'
    elif state=='ML':
        sname='Meghalaya'
    elif state=='MN':
        sname='Manipur'
    elif state=='MZ':
        sname='Mizoram'
    elif state=='NL':
        sname='Nagaland'
    elif state=='OR':
        sname='Orissa'
    elif state=='PB':
        sname='punjab'
    elif state=='RJ':
        sname='rajasthan'
    elif state=='SK':
        sname='Sikkim'
    elif state=='TN':
        sname='Tamil_Nadu'
    elif state=='TG':
        sname='telangana'
    elif state=='TR':
        sname='Tripura'
    elif state=='UT':
        sname='uttarakhand'
    elif state=='UP':
        sname='uttar_pradesh'
    elif state=='WB':
        sname='Westbengal'
    elif state=='CH':
        sname='Chandigarh'
    elif state=='DL':
        sname='delhi'    
    print(sname)
    
    state = "data/" + sname + ".xlsx"
    generate(state)
    print(state)
    final=""
    if power=='biomass':
        final="Predicted Biomass Power"
        res=BiomassPredict(state)
    elif power=='wind':
        final="Predicted Wind Power"
        res=windPredict(state)
    elif power=='solar':
        final="Predicted Solar Power"
        res=solarPredict(state)
    elif power=='smallhydro':
        final="Predicted Small Hydro Power"
        res=smallHydroPredict(state)
        #print("res is ",res)
    res=res.tolist()
    print(res)
    print(type(res))
    for i in range(len(res)):
        res[i]=round(res[i],4)
    return render_template('predictpower.html',final=final,result=res)

app.run(debug=True)