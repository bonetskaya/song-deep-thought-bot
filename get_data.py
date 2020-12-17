from __future__ import unicode_literals
import youtube_dl
import os
import glob

playlist_url = ""

os.system('!youtube-dl --dump-json --flat-playlist "' + playlist_url + '" \  | jq
        -r '"\(.title)\nhttps://youtu.be/\(.id)\n"' > playlist.txt')

with open('playlist.txt', 'r') as f:
    for i, url in enumerate(f):
        if i % 3 != 1:
            continue
        j = (i - 1) // 3
        ydl_opts = {
        'outtmpl': './' + str(j) +'.mp4',
        'format': '(bestvideo[width>=1080][ext=mp4]/bestvideo)+bestaudio/best',
        'writesubtitles': True,
        'subtitle': '--write-sub --sub-lang en',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download Successful!")
        if len(glob.glob("7/" + str(j) + ".*en*.vtt")) == 0:
            os.system("rm 7/" + str(j) +'.*')
os.system('find . -type f -name "*.mkv" -exec bash -c 'FILE="$1"; ffmpeg -i "${FILE}" -vn -c:a libmp3lame -y "${FILE%.mkv}.mp3";' _ '{}' \;')
