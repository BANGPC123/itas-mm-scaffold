export function createVideoSpeech({synthesize,play,onError=()=>{},cooldown=20,maxAge=5}) {
  let epoch=0,busy=false,retryAfter=-Infinity;
  const spoken=new Map();
  return {
    reset(){epoch++;busy=false;retryAfter=-Infinity;spoken.clear();},
    async update(text,timestamp,currentTime){
      if(!text||busy||timestamp<retryAfter||timestamp-(spoken.get(text)??-Infinity)<cooldown)return;
      const own=epoch;busy=true;
      try{
        const url=await synthesize(text);
        if(own!==epoch||currentTime()<timestamp||currentTime()-timestamp>maxAge)return;
        spoken.set(text,timestamp);
        await play(url);
      }catch(error){if(own===epoch){retryAfter=currentTime()+5;onError(error);}}
      finally{if(own===epoch)busy=false;}
    }
  };
}
