from ultralytics import YOLO
import os
import click
import yaml

@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('train_images_path', nargs=1, type=click.Path())
@click.argument('val_images_path', nargs=1, type=click.Path())
@click.argument('data_yaml_path', nargs=1, type=click.Path())
@click.option('--model', '-m', 'model', type=str, default='yolov8n.pt')
@click.option('--coco-file-path', '-c', 'coco_file_path', type=click.Path(), default='coco.txt')
@click.option('--resume/--no-resume', '-R/-r', 'resume_training', is_flag=True, default=True)
@click.option('--project', '-p', 'project', type=str)
@click.option('--name', '-n', 'name', type=str)
@click.option('--epochs', type=int, default=100)
@click.option('--imgsz', type=int, default=512)
@click.option('--batch', type=int, default=32)
@click.option('--workers', type=int, default=8)
@click.option('--patience', type=int, default=0)
@click.option('--cache', is_flag=True)
@click.option('--single-cls/--multiple-cls', is_flag=True, default=True)
@click.option('--mosaic', type=float, default=1.0)
@click.option('--mixup', type=float, default=0.2)
@click.option('--degrees', type=float, default=6.0)
@click.option('--translate', type=float, default=0.1)
@click.option('--scale', type=float, default=0.5)
@click.option('--shear', type=float, default=0.1)
@click.option('--flipud', type=float, default=0.1)
@click.option('--fliplr', type=float, default=0.5)
@click.option('--close-mosaic', type=int, default=30)
@click.option('--multi-scale/--single-scale', is_flag=True, default=True)
def main(**options):
  # === Generate YOLO data config ===

  data_content = {
     'path':'.',
     'train':options['train_images_path'],
     'val':options['val_images_path'],
     'names':[]
  }

  with open(options['coco_file']) as coco_file:
     pass

  with open(options['data_yaml_path'], "w") as f:
      yaml.dump(data_content, options['data_yaml_path'], default_flow_style=False)

  project = options['project']
  name = options['name']

  if not project:
     project = 'yolo'
  if not name:
     name = 'iteration-1'

  # === Build checkpoint path dynamically from project and name ===
  checkpoint_path = os.path.join(project, name, "weights", "last.pt")

  # === Control resume training with a simple flag ===
  resume_mode = options['resume_training'] and os.path.exists(checkpoint_path)

  # === Load model: resume from checkpoint if available, otherwise start fresh ===
  model_path = checkpoint_path if resume_mode else model  # Replace with your preferred model file if needed
  model = YOLO(model_path)

  # === Train the model ===
  model.train(
      data=options['data_yaml_path'],
      epochs=options['epochs'],
      imgsz=options['imgsz'],
      batch=options['batch'],
      workers=options['workers'],
      patience=options['patience'],           # Set to 0 to disable early stopping (beware, patience only looks at accuracy, and is unaware of any improvements in loss)
      cache=options['cache'],
      single_cls=options['single_cls'],
      mosaic=options['mosaic'],
      mixup=options['mixup'],
      degrees=options['degrees'],          # Small rotations (in degrees) for augmentation
      translate=options['translate'],        # Shifts objects up to 10% of the image dimensions
      scale=options['scale'],            
      shear=options['shear'],            # Apply a shear transformation with a factor of 0.1
      flipud=options['flipud'],           
      fliplr=options['fliplr'],           # Horizontal flip probability
      close_mosaic=options['close_mosaic'],      # Disables mosaic augmentation 30 epochs before training ends (with epochs=100, mosaic stops at epoch 70)
      multi_scale=options['multi_scale'],     # Enable multi-scale training for better detection of objects of varied sizes
      project=project,
      name=name,
      resume=resume_mode,
  )

if __name__ == '__main__':
    main()
