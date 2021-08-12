from api.mlcode.model import StyleTransfer

def train_and_get_result(urls):
    style_transfer_class = StyleTransfer(
    base_image_name='base.png',
    style_image_name='style.png',
    base_image_url=str(urls['structure']),
    style_image_url=str(urls['style'])
    )
    iterations = urls['iters']

    style_transfer_class.setupModel(iterations=int(iterations))

    style_transfer_class.train()
    
    #get the outputs as image urls and return them
    return(f'generated_at_iteration_{iterations}.png')
