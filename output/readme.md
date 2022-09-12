Here are the examples to reproduce the toonification results in this folder.

#### 529_vtoonify_d.mp4
https://user-images.githubusercontent.com/18130694/189573234-c2a19208-e93a-497f-ad90-a6fb942543d2.mp4

```python
python style_transfer.py --scale_image --content ./data/529.mp4 --video
```
<br/>

#### 077436_vtoonify_d.jpg
<img src="./077436_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --scale_image
```
<br/>

#### 081680_vtoonify_d.jpg
<img src="./081680_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --scale_image --content ./data/081680.jpg \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
```
<br/>

#### 038648_vtoonify_d.jpg
<img src="./038648_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt \
```
<br/>

#### 038648_vtoonify_t.jpg
<img src="./038648_vtoonify_t.jpg" width=25%>

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify.pt \
```
<br/>

#### 077559_vtoonify_d.jpg
<img src="./077559_vtoonify_d.jpg" width=25%>

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600 --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt \
```
<br/>

#### 077559_vtoonify_t.jpg
<img src="./077559_vtoonify_t.jpg" width=25%>

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify.pt \
```
