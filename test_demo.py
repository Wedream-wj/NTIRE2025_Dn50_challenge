import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # SGN test
        from models.team00_SGN import SGNDN3
        name, data_range = f"{model_id:02}_RFDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team00_sgn.ckpt')
        model = SGNDN3()

        state_dict = torch.load(model_path)["state_dict"]
        state_dict.pop("current_val_metric")
        state_dict.pop("best_val_metric")
        state_dict.pop("best_iter")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.find("model.") >= 0:
                new_state_dict[k.replace("model.", "")] = v
        model.load_state_dict(new_state_dict, strict=True)

    elif model_id == 6:
        # NAFNet+TLC+TTA
        from models.team06_NAFNet import NAFNetLocal
        name, data_range = f"{model_id:02}_NAFNetLocal", 1.0
        model_path = os.path.join('model_zoo', 'team06_NAFNet.pth')
        model = NAFNetLocal( train_size=(1, 3, 384, 384),
                   img_channel=3,
                   width=64,
                   middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8],
                   dec_blk_nums=[2, 2, 2, 2])

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        # RCAN+TTA
        # from models.team06_RCAN import RCAN
        # name, data_range = f"{model_id:02}_RCAN", 1.0
        # model_path = os.path.join('model_zoo', 'team06_RCAN.pth')
        # model = RCAN(num_in_ch=3, num_out_ch=3, num_feat=64, num_group=10, num_block=20, \
        #          squeeze_factor=16, upscale=1, res_scale=1)
        #
        # loadnet = torch.load(model_path)
        # if 'params_ema' in loadnet:
        #     keyname = 'params_ema'
        # else:
        #     keyname = 'params'
        # model.load_state_dict(loadnet[keyname], strict=True)


    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    elif mode == "valid":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_valid_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]
    elif mode == "hybrid_test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "noise50.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=1):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_noisy, img_hr) in enumerate(data_path):
        # print(img_noisy)
        # print(img_hr)
        # --------------------------------
        # (1) img_noisy
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_noisy = util.imread_uint(img_noisy, n_channels=3)
        # print(img_noisy.shape)
        img_noisy = util.uint2tensor4(img_noisy, data_range)
        img_noisy = img_noisy.to(device)

        # --------------------------------
        # (2) img_dn
        # --------------------------------
        start.record()
        img_dn = forward(img_noisy, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_dn = util.tensor2uint(img_dn, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_dn.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_dn, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_dn, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_dn_y = util.rgb2ycbcr(img_dn, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_dn_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_dn_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
        util.imsave(img_dn, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2025-Dn50", log_path="NTIRE2025-Dn50.log")
    logger = logging.getLogger("NTIRE2025-Dn50")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        if args.hybrid_test:
            # inference on the DIV2K and LSDIR test set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="hybrid_test")
            # record PSNR, runtime
            results[model_name] = valid_results
        else:
            # inference on the validation set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
            # record PSNR, runtime
            results[model_name] = valid_results

            if args.include_test:
                # inference on the test set
                test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
                results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        flops = get_model_flops(model, input_dim, False)
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        # print(v.keys())
        if args.hybrid_test:
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-Dn50")
    parser.add_argument("--data_dir", default="./NTIRE2025_Challenge/input", type=str)
    parser.add_argument("--save_dir", default="./NTIRE2025_Challenge/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--hybrid_test", action="store_true", help="Hybrid test on DIV2K and LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
