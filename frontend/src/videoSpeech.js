// queueWait > 0 keeps one cue that arrives while another is speaking, for up to queueWait video seconds.
export function createVideoSpeech({synthesize,play,onError=()=>{},cooldown=20,maxAge=5,queueWait=0}) {
  let epoch=0,busy=false,retryAfter=-Infinity,waiting=null;
  const spoken=new Map();
  async function speak(text,timestamp,currentTime){
    const own=epoch;busy=true;
    try{
      const url=await synthesize(text);
      if(own!==epoch||currentTime()<timestamp||currentTime()-timestamp>maxAge)return;
      spoken.set(text,timestamp);
      await play(url);
    }catch(error){if(own===epoch){retryAfter=currentTime()+5;onError(error);}}
    finally{
      if(own===epoch){
        busy=false;
        const next=waiting;waiting=null;
        if(next&&currentTime()-next.queuedAt<=queueWait)speak(next.text,next.timestamp,currentTime);
      }
    }
  }
  return {
    reset(){epoch++;busy=false;retryAfter=-Infinity;waiting=null;spoken.clear();},
    update(text,timestamp,currentTime){
      if(!text||timestamp<retryAfter||timestamp-(spoken.get(text)??-Infinity)<cooldown)return;
      if(busy){
        if(queueWait>0&&!waiting&&text!==waiting?.text)waiting={text,timestamp,queuedAt:currentTime()};
        return;
      }
      return speak(text,timestamp,currentTime);
    }
  };
}
