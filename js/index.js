// import BotUI from 'botui';
var botui = new BotUI('help-bot') // id of container

//Function to query flask api for dialogflow response
const callAPI = async (requestStr,history,task_id,lang_id,top_p,temperature,repetition,lon,lat) => {
    //request = btoa(`${history}______________${requestStr}______________${task_id}______________${lang_id}______________${top_p}______________${temperature}______________${repetition}______________${lon}______________${lat}`)
    // console.log(request);
    let body = {
        'history': history,
        'query': requestStr,
        'task_name': task_id,
        'lang_id': lang_id,
        'top_p': top_p,
        'temp': temperature,
        'rept': repetition,
        'lon': lon,
        'lat': lat
    };
    let options = {
        method: 'POST',
        body: JSON.stringify(body),
        headers: {
            'Content-Type': 'application/json'
        }
    }
    const response = await fetch('/request', options);
    const myJson = await response.json(); 
    return (myJson);

}

botui.message.bot({ // show first message
    delay: 200,
    content: 'Hi, I am a good talker. We can chat about anything you want '
})

var dialogue_history = []
var task_id = -1
var language_id = "en"
var lat = 0
var lon = 0
// if (window.navigator && window.navigator.geolocation) {
//     navigator.geolocation.getCurrentPosition(onGeolocateSuccess, onGeolocateError);
// }

function replace_day(day){
    var day_to_abbv = {
        Monday: 'Mon',
        Tuesday: 'Tue',
        Wednesday: 'Wed',
        Thursday:'Thur',
        Friday: 'Fri',
        Saturday: 'Sat',
        Sunday: 'Sun'
    }
    return day_to_abbv[day];
}

function get_whether_imgsrc(weather){
    var weather_to_src = {
        Thunderstorm : '../images/weather_icons/tstorms.png',
        Drizzle: '../images/weather_icons/chancerain.png',
        Rain: '../images/weather_icons/rain.png',
        Snow:'../images/weather_icons/snow.png',
        Mist: '../images/weather_icons/atmosphere.png',
        Smoke: '../images/weather_icons/atmosphere.png',
        Haze: '../images/weather_icons/atmosphere.png',
        Dust: '../images/weather_icons/atmosphere.png',
        Fog: '../images/weather_icons/atmosphere.png',
        Sand: '../images/weather_icons/atmosphere.png',
        Ash: '../images/weather_icons/atmosphere.png',
        Squall: '../images/weather_icons/atmosphere.png',
        Tornado: '../images/weather_icons/atmosphere.png',
        Clear: '../images/weather_icons/sunny.png',
        Clouds: '../images/weather_icons/cloudy.png'
    }
    return weather_to_src[weather];
}

function addWeatherTable(weather_matrix, city) {
  var myTableDiv = document.getElementById("weather_info_");
  var table_exist = document.getElementById("weather_table");

  if (table_exist != null){
      table_exist.remove();
  }

    var location = document.createElement("H6");
    var t = document.createTextNode("\u00A0\u00A0\u00A0\u00A0" +city);     // Create a text node
    location.appendChild(t);

  var table = document.createElement('TABLE');
  table.className = 'table';
  table.id = 'weather_table'

  var tableBody = document.createElement('TBODY');
  table.appendChild(tableBody);

  for (var i = 0; i < 3; i++) {
    var tr = document.createElement('TR');
    tableBody.appendChild(tr);

    if( i == 0){
        var td = document.createElement('TH');
          td.appendChild(document.createTextNode('Today'));
          tr.appendChild(td);

        for (var j = 1; j < 5; j++) {
          var td = document.createElement('TH');
          td.appendChild(document.createTextNode(weather_matrix[0][j]));
          tr.appendChild(td);
        }
    }else if ( i == 2 ){
        for (var j = 0; j < 5; j++) {
          var td = document.createElement('TD');
          td.appendChild(document.createTextNode(weather_matrix[2][j]));
          tr.appendChild(td);
        }
    }
    else{
        for (var j = 0; j < 5; j++) {
          var td = document.createElement('TD');
          var weather_img = document.createElement("img");
            weather_img.src = get_whether_imgsrc(weather_matrix[1][j]);
            weather_img.style.width = "70%";
            weather_img.style.height = "auto";
          td.appendChild(weather_img);
          tr.appendChild(td);
        }
    }
  }
  // myTableDiv.appendChild(location)
  myTableDiv.appendChild(table);
}

function display_weather(weather_data){

    if (weather_data == undefined || weather_data.length == 0) {
        document.getElementById("weather_info_").style.display='none';
        console.log("no weather data");
    }
    else{
        document.getElementById("weather_info_").style.display='block';
        var matrix = Array.from(Array(3), () => new Array(5));
        for(let i = 1; i < weather_data.length - 1; i++) {
          matrix[0][i-1] = replace_day(weather_data[i][1]);
          matrix[1][i-1] = weather_data[i][2];
          matrix[2][i-1] = weather_data[i][3] +' / '+ weather_data[i][4];
        }
        addWeatherTable(matrix, weather_data[1][0]);
    }
}


function display_wiki(wiki_data) {
    if (wiki_data == ""){
        return;
    }
  var myTableDiv = document.getElementById("wiki_result");
  var table_exist = document.getElementById("wiki_table");

  if (table_exist != null){
      table_exist.remove();
  }

  var table = document.createElement('TABLE');
  table.className = 'table';
  table.id = 'wiki_table'

  var tableBody = document.createElement('TBODY');
  table.appendChild(tableBody);

    var tr = document.createElement('TR');
    tableBody.appendChild(tr);

    var th = document.createElement('TH');
      th.appendChild(document.createTextNode("From Wikipedia"));
      tr.appendChild(th);

    var td = document.createElement('TD');
    td.appendChild(document.createTextNode(wiki_data));
    td.id = "wiki_cell";
    tr.appendChild(td);

  myTableDiv.appendChild(table);
}

function init() {
    botui.action.text({
        action: {
            placeholder: "Ask me anything..."
        }
    }).then(async (res) => {
        botui.message.add({
            loading: true
        }).then(async (index) => {
            // console.log('TASK ID', task_id);
            temperature = document.getElementById("temp_slide").value;
            top_p = document.getElementById("top-p_slide").value;
            repetition = document.getElementById("repetition_slide").value;
            // dialogue_history.push(res.value);
            jsonResult = await callAPI(res.value,dialogue_history,task_id,language_id,top_p,temperature,repetition,lat,lon);
         // 1. Attaching Emoji to user input
            var y = document.getElementsByClassName("botui-message");

            var target = y[y.length-2];
            var emoji_fontSize = 25;
            const bottom = 24; //need to calculate with padding and fontsize later
            var width = target.offsetWidth - (emoji_fontSize+2);
            var height = target.offsetHeight - (emoji_fontSize * 0.33);

            const menu = document.querySelector('#help-bot');
            let element = document.createElement('div');

            // element.innerHTML = '<p>&#128150;</p>';
            element.innerHTML = '<p>' + jsonResult['user_emoji'] + '</p>';
            element.style.position = "relative";
            element.style.fontSize = emoji_fontSize + 'px';
            element.style.left = width + 'px';
            element.style.top = height  + 'px';
            element.style.zIndex = '100';
            y[y.length-2].appendChild(element);

            if(task_id=="AUTOMODE"){
                document.getElementById("weather_info_").style.display='none';
                document.getElementById("kb_graph").style.display='none';
                document.getElementById("wiki_result").style.display='none';
            }

         // 2. If the task is weather
         //     if (task_id == 'weather'){
             if (jsonResult['task_name']== "SMD"){
                display_weather(jsonResult['viz_meta']['Wea']);
             }
         // 3. If the task is Wiki
         //    if (task_id == 'WoW'){
            if(jsonResult['task_name']== "WoW") {
                if (jsonResult['viz_meta']['Wiki']!= null){
                    document.getElementById("wiki_result").style.display='block';
                    display_wiki(jsonResult['viz_meta']['Wiki'])
                }
            }
         // 4. If the task is [ "Movie","Book","Music","Sport"] kb graph
            var kb_graph_tasks = [ "Movie","Book","Music","Sport"];
            // if(kb_graph_tasks.includes(task_id)) {
            if(jsonResult['task_name']== "dialKG") {
                if (jsonResult['viz_meta']['graph'] != null){
                    document.getElementById("kb_graph").style.display='block';
                    var json_data = JSON.parse(jsonResult['viz_meta']['graph']);
                    var neo4jd3 = new Neo4jd3('#kb_graph', {
                        icons: {},
                        images: {},
                        minCollision: 60,
                        neo4jData: json_data,
                        // neo4jDataUrl: '../json/neo4jData.json', //neo4jData : Graph data in Neo4j data format.
                        nodeRadius: 30,
                        zoomFit: false
                    });
                }else{
                    document.getElementById("kb_graph").style.display='none';
                }
            }
            dialogue_history = jsonResult['history']
            return botui.message.update(index, {
                id: 1,
                loading: false,
                content: `${jsonResult['response']} ${jsonResult['resp_emoji']} `
            });
        }).then(init); //ask again, and keep it in loop
    })
}


//=========================================================================================
function key_custom(task_name) {
    // console.log(task_name);
    if(task_name != task_id){dialogue_history = []}
    task_id = task_name
    if(task_name=="CovidQA"){
        botui.message.bot({ // show first message
            delay: 200,
            content: 'Ask a question about COVID-19'
        })
     }
    if(task_name=="debunker"){
        botui.message.bot({ // show first message
            delay: 200,
            content: 'Give me a claim about COVID-19 and I will debunk it for you'
        })
    }
    if (task_name != "AUTOMODE"){
        for (const ts of TASKS_NAMES_IN_MENU) {
            if(ts != task_name && document.getElementById(ts)!= null){
                document.getElementById(ts).className = "btn btn-outline-primary btn-sm";
            }   
        }
        document.getElementById(task_name).className = "btn btn-primary btn-sm";
    }

    if (task_name !="weather"){
        document.getElementById("weather_info_").style.display='none';
    }
    document.getElementById("kb_graph").style.display='none';
    document.getElementById("wiki_result").style.display='none';

    var temp_custom = [ "anger","fear","joy","sadness","surprised"];

    if(temp_custom.includes(task_name)){
            temperature = 0.3;
            document.getElementById("temp_slide").value = 0.3;
    }else{
        if(task_name == "caire"){
            temperature = 0.7;
            document.getElementById("temp_slide").value = 0.7;
        }
        else{
            temperature = 0.6;
            document.getElementById("temp_slide").value = 0.6;
        }
    };
    return ;
}


function set_language(id) {
    language_id = id
    console.log(language_id)
    // var lang_array = ["en", "zh", "it", "id", "fr", "es", "de"]
    // lang_array.forEach(item =>  document.getElementById(item).className = "btn btn-outline-primary btn-lang btn-sm");
    // document.getElementById(language_id).className = "btn btn-primary btn-lang btn-sm";

    return ;
}

TASKS_NAMES_IN_MENU = [
    "caire","persona","dialGPT",
    "anger","fear","joy","sadness",
    "surprised","positive","negative",
    "question","business","sport","sci_tech",
    // "MWoZ_attraction","MWoZ_taxi","MWoZ_hotel",
    // "MWoZ_restaurant","MWoZ_train",
    "weather", "navigate","schedule","Movie","Book",
    "Music","Sport","WoW","CovidQA","debunker"
]

// key_custom("AUTOMODE")
key_custom("caire")
set_language("en")
init()

function collapse_btns(btn){
    var isExpanded = $('#manualBtns').hasClass("show");
    // document.getElementById("kb_graph").style.display='none';

    if(btn.id=='auto_btn'){
        if(isExpanded){
            $('#manualBtns').collapse({toggle: false}).collapse('hide');
        }
        // document.getElementById("manual_btn").style.backgroundColor = "#6c757d";
        document.getElementById("manual_btn").className = "btn btn-primary btn-sm";
        document.getElementById("manual_btn").style.opacity = 0.5;

    }
    if(btn.id=='manual_btn'){
        if(!isExpanded){
            for (i = -1; i < 24; i++) {
                if(document.getElementById(i) != null){
                    document.getElementById(i).className = "btn btn-outline-primary btn-sm";
                }
            }
            $('#manualBtns').collapse({toggle: true}).collapse('show');

        }
        document.getElementById("auto_btn").className = "btn btn-primary btn-sm";
        document.getElementById("auto_btn").style.opacity = 0.5;

    }
    btn.className = "btn btn-primary btn-sm";
    btn.style.opacity = 1;

}

// collapse_btns(document.getElementById('auto_btn'))
collapse_btns(document.getElementById('manual_btn'))



// function foo() {

//     Webcam.snap( function(data_uri) {

//         var xhr = new XMLHttpRequest();
//         var url = "https://eez114.ece.ust.hk:8080/faceEmo";
//         xhr.open("POST", url, true);
//         xhr.setRequestHeader("Content-Type", "application/json");
//         xhr.onreadystatechange = function () {
//             if (xhr.readyState === 4 && xhr.status === 200) {
//                 var json = JSON.parse(xhr.responseText);
//                 console.log(json);
//             }
//         };
//         var data = JSON.stringify({"photo": data_uri});
//         xhr.send(data);
//     } );

//     setTimeout(foo, 1000);
// }

// foo();

var i = 0;
function background_change(emotion_json) {
  var doc = document.getElementById("page-container");

  var testing = ["anger", "happiness", "neutral", "sadness", "surprise",  "neutral", "sadness"];
//   let colormatch = { anger: "rgba(235,80,71,0.79)", happiness: "rgba(232,217,63,0.73)" , neutral: "white",
//       sadness: "rgba(42,74,156,0.67)", surprise: "rgba(74,202,232,0.67)", disgusted:"rgba(127,39,8,0.67)",
//       fearful:"rgba(27,29,23,0.67)"};
  let colormatch = { anger: "rgba(250,110,89,0.79)", happiness: "rgba(255,219,92,0.73)" , neutral: "white",
      sadness: "rgba(42,74,156,0.67)", surprise: "rgba(72,151,218,0.67)", disgusted:"rgba(127,39,8,0.67)",
      fearful:"rgba(27,29,23,0.67)"};
  doc.style.backgroundColor = colormatch[testing[i]]
  i = (i + 1) % testing.length;
}
// setInterval(background_change, 2000);

function resetHeight(){
    var newHeight = $(window).height() - $('#footer').outerHeight();
    $('.botui-app-container').each(function() {
       $(this).height(newHeight);
     });

}
$(window).resize(function(){
    resetHeight();
});

// function onGeolocateSuccess(coordinates) {
//     const { latitude, longitude } = coordinates.coords;
//     // console.log('Found coordinates: ', latitude, longitude);
//     lat = longitude
//     lon = latitude
//   }
  
// function onGeolocateError(error) {
// console.warn(error.code, error.message);

// if (error.code === 1) {
//     // they said no
//     lat = 0
//     lon = 0
// } else if (error.code === 2) {
//     lat = 0
//     lon = 0
//     // position unavailable
// } else if (error.code === 3) {
//     lat = 0
//     lon = 0
//     // timeout
// }
// }

