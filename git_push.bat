@echo off
echo 開始推送 Git 專案...
git add .
git commit -m "Auto commit from bat file"
git push
pause