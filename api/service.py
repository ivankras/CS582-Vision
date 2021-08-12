from mlcode.model import StyleTransfer
import shutil, os

def train_and_get_result(urls):
    if 'iters' not in urls.keys():
        urls['iters'] = 400
    style_transfer_class = StyleTransfer(
    base_image_name=str(urls['structure'].split('/')[-1] + '.jpeg'),
    style_image_name=str(urls['style'].split('/')[-1] + '.jpeg'),
    base_image_url=str(urls['structure']),
    style_image_url=str(urls['style'])
    )
    if os.path.exists('api/imgs'):
        shutil.rmtree('api/imgs')
    os.mkdir('api/imgs')
    iterations = urls['iters']

    style_transfer_class.setupModel(iterations=int(iterations))

    style_transfer_class.train()
    del style_transfer_class
    
    return(f'imgs/generated_at_iteration_{iterations}.png')
