Here are the examples to reproduce the toonification results in this folder.

#### 529_vtoonify_d.mp4
https://user-images.githubusercontent.com/18130694/189574934-85954bd6-0e2d-4568-9687-df7a8776ace5.mp4

```python
python style_transfer.py --scale_image --content ./data/529.mp4 --video
```
<br/>

#### 077436_vtoonify_d.jpg
<img src="./077436_input.jpg" width=25%> &nbsp; <img src="./077436_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --scale_image
```
<br/>

#### 081680_vtoonify_d.jpg
<img src="./081680_input.jpg" width=25%> &nbsp; <img src="./081680_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --scale_image --content ./data/081680.jpg \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt # specialized model has better performance
```
<br/>

#### 038648_vtoonify_d.jpg
<img src="./038648_input.jpg" width=25%> &nbsp; <img src="./038648_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt 
```
<br/>

#### 038648_vtoonify_t.jpg
<img src="./038648_input.jpg" width=25%> &nbsp; <img src="./038648_vtoonify_t.jpg" width=25%>

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify.pt 
```
<br/>

#### 077559_vtoonify_d.jpg
<img src="./077559_input.jpg" width=25%> &nbsp; <img src="./077559_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600 --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt 
```
<br/>

#### 077559_vtoonify_t.jpg
<img src="./077559_input.jpg" width=25%> &nbsp; <img src="./077559_vtoonify_t.jpg" width=25%>

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify.pt 
```
