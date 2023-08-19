install_req:
	@-pip install --upgrade pip
	@-pip install -r requirements.txt

run:
	streamlit run app.py

env_load:
	direnv allow .
	direnv reload

first:
	make install_req
	make env_load
	make run

all:
	make env_load
	make run
push:
	git add .
	git commit -m "update"
	git push origin master
