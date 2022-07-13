import dominate
from dominate.tags import *
import os, json, datetime


class HTML:
    def __init__(self, web_dir, exp_name, config, title='seg results', save_name='index', reflesh=0, resume=None):
        self.title = title
        self.web_dir = web_dir
        self.save_name = save_name + '.html'

        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        html_file = os.path.join(self.web_dir, self.save_name)

        if resume is not None and os.path.isfile(html_file):
            self.old_content = open(html_file).read()
        else:
            self.old_content = None

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        header = f'Experiment name: {exp_name}, Date: {date_time}'
        self.add_header(header)
        self.add_header('Configs')
        self.add_config(config)
        with self.doc:
            hr()
            hr()
        self.add_table()

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_config(self, config):
        t = table(border=1, style="table-layout: fixed;")
        self.doc.add(t)
        conf_model = config['model']
        with t:
            with tr():
                with td(style="word-wrap: break-word;", halign="center", valign="top"):
                    td(f'Epochs : {config["trainer"]["epochs"]}')
                    td(f'Lr scheduler : {config["lr_scheduler"]}')
                    td(f'Lr : {config["optimizer"]["args"]["lr"]}')
                    if "datasets" in list(config.keys()): td(f'Datasets : {config["datasets"]}')
                    td(f"""Decoders : Vat {conf_model["vat"]} Dropout {conf_model["drop"]} Cutout {conf_model["cutout"]}
                                     FeatureNoise {conf_model["feature_noise"]} FeatureDrop {conf_model["feature_drop"]}
                                     ContextMsk {conf_model["context_masking"]} ObjMsk {conf_model["object_masking"]}""")
        if "datasets" in list(config.keys()):
            self.doc.add(p(json.dumps(config[config["datasets"]], indent=4, sort_keys=True)))
        else:
            self.doc.add(p(json.dumps(config["train_supervised"], indent=4, sort_keys=True)))

    def add_results(self, epoch, seg_resuts, width=400, domain=None):
        para = p(__pretty=False)
        with self.t:
            with tr():
                with td(style="word-wrap: break-word;", halign="center", valign="top"):
                    td(f'Epoch : {epoch}')
                    if domain is not None:
                        td(f'Mean_IoU_{domain} : {seg_resuts[f"Mean_IoU_{domain}"]}')
                        td(f'PixelAcc_{domain} : {seg_resuts[f"Pixel_Accuracy_{domain}"]}')
                        td(f'Val Loss_{domain} : {seg_resuts[f"val_loss_{domain}"]}')
                    else:
                        td(f'Mean_IoU : {seg_resuts["Mean_IoU"]}')
                        td(f'PixelAcc : {seg_resuts["Pixel_Accuracy"]}')
                        td(f'Val Loss : {seg_resuts["val_loss"]}')

    def save(self):
        html_file = os.path.join(self.web_dir, self.save_name)
        f = open(html_file, 'w')
        if self.old_content is not None:
            f.write(self.old_content + self.doc.render())
        else:
            f.write(self.doc.render())
        f.close()
