# Face-Match
### 
1. Clone the repo.
  
2. install the dependencies:-
```
pip install -r requirements.txt
```

3. run the command to run the application:-
```
uvicorn face-match-api:app --reload
```
4. Use the curl command to test the application:-
   A. /generate-encodings
   ```
   curl -X POST http://127.0.0.1:8000/generate-encodings
   ```
   B. /upload-and-match
   ```
   curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:8000/upload-and-match
   ```

   
