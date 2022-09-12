Here are the examples to reproduce the toonification results in this folder.

#### 529_vtoonify_d.mp4

```python
python style_transfer.py --scale_image --content ./data/529.mp4 --video
```

#### 077436_vtoonify_d.jpg

```python
python style_transfer.py --scale_image
```

#### 081680_vtoonify_d.jpg

```python
python style_transfer.py --scale_image --content ./data/081680.jpg \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
```

#### 038648_vtoonify_d.jpg

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600  --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt \
```

#### 038648_vtoonify_t.jpg

```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify_s_d.pt \
```

#### 077559_vtoonify_d.jpg

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600  --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt \
```

#### 077559_vtoonify_t.jpg

```python
python style_transfer.py --content ./data/077559.jpg \
       --scale_image --padding 600 600 600 600 --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify_s_d.pt \
```
