import time

import torch
import torch.nn as nn

from quip.gptq import *
from quip.bal import Balance
from quip.near import Nearest
from quip.modelutils import *
from quip.quant import *

from tqdm import tqdm


def get_opt(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from ringrwkv.configuration_rwkv_world import RwkvConfig
    from ringrwkv.modehf_world import RwkvForCausalLM

    model = RwkvForCausalLM.from_pretrained("StarRing2022/RWKV-4-World-1.5B")
    model.seqlen = 1024
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.rwkv.blocks

    # model.rwkv.embeddings = model.rwkv.embeddings.to(dev)
    # Set all LayerNorm layer to device
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.to(dev)
    # model.rwkv.ln_out = model.rwkv.ln_out.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # Should infer once to turn off self.layers_are_rescaled and not make Catcher crash
    for batch in dataloader:
        model(batch[0].to(dev))
        break

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            # TODO: add back attention_mask and inputs_ids
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.cpu()
    # model.rwkv.ln_out = model.rwkv.ln_out.cpu()
    # layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    print("Ready.")

    # QUIP PATH
    quantizers = {}
    errors, Hmags, times = [], [], []
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        # Initialize Quant Method and Compute H
        for name in subset:
            if args.quant == "gptq":
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False
                )
            elif args.quant == "nearest":
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False
                )
            elif args.quant in ["allbal", "ldlq", "ldlqRG", "ldlbal_admm"]:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                    args.quant, args.wbits, args.npasses, unbiased=args.unbiased
                )
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, qfn=args.qfn, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        # (H / nsamples).to(torch.float32)
        for name in subset:
            quant_method[name].post_batch()

        # Quantize Weights
        for name in subset:
            print(i, name)
            print("Quantizing ...")
            quant_method[name].preproc(
                preproc_gptqH=args.pre_gptqH,
                percdamp=args.percdamp,
                preproc_rescale=args.pre_rescale,
                preproc_proj=args.pre_proj,
                preproc_proj_extra=args.pre_proj_extra,
            )
            if args.quant == "gptq":
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ["allbal", "ldlq", "ldlqRG", "ldlbal_admm"]:
                quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
            elif args.quant == "nearest":
                quant_method[name].fasterquant()
            quantizers["model.decoder.layers.%d.%s" % (i, name)] = quant_method[
                name
            ].quantizer

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # print("errors")
    # print(errors)
    # print("Hmags")
    # print(Hmags)
    print(f"Total quant time: {sum(times):.2f}s")

    return quantizers, errors


def benchmark(model, testloader, benchmark_samples):
    ctx_len = 1024
    stride = ctx_len // 2
    seq_len = testloader.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, stride * benchmark_samples, stride)):
        end_loc = min(begin_loc + ctx_len, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = testloader.input_ids[:, begin_loc:end_loc].to(DEV)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print(f"nlls: {torch.stack(nlls)}")
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="OPT model to load; pass `facebook/opt-X`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--quant",
        choices=["allbal", "ldlq", "ldlqRG", "ldlbal_admm", "nearest", "gptq"],
        default="nearest",
        help="Which quantization method to use.",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--npasses",
        type=int,
        default=0,
        help="number passes to repeat balance loop over 1-d.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument("--pre_gptqH", action="store_true", help="preprocessing")
    parser.add_argument("--pre_rescale", action="store_true", help="preprocessing")
    parser.add_argument("--pre_proj", action="store_true", help="preprocessing")
    parser.add_argument(
        "--pre_proj_extra",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Extra options to control pre_proj step.",
    )
    parser.add_argument(
        "--qfn",
        type=str,
        default="a",
        help="qfn: a is default, b is sym incoherent based",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    # parser.add_argument('--benchmark',
    #                     type=int,
    #                     default=0,
    #                     help='Number of tokens to use for benchmarking.')
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--proxy_only",
        action="store_true",
        help="Only compute proxy objective (w^T H w)",
    )
    parser.add_argument("--unbiased", action="store_true", help="unbiased")
    parser.add_argument(
        "--incoh_processing", action="store_true", help="incoherence processing"
    )
    parser.add_argument(
        "--lazy_batch",
        action="store_true",
        help="lazy batch updates in blocks as used in OPTQ",
    )

    args = parser.parse_args()
    # defaults to incoherence processing
    if args.incoh_processing:
        args.pre_gptqH = True
        args.pre_rescale = True
        args.pre_proj = True
        args.proj_extra = 0
        args.qfn = "b"

    if args.load:
        model = load_quant(args.model, args.load)
        model.eval()
    else:
        model = get_opt(args.model)
        model.eval()

        dataloader, _ = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

        if args.wbits < 16:
            # Preprocessing flags
            if args.qfn == "b":
                assert args.pre_proj is True
            print(
                f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}"
            )
            print(f"using lazy_batch updates: {args.lazy_batch}")
            # LDL checks
            if ("ldl" in args.quant) and args.unbiased and (args.npasses > 0):
                print(
                    f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED."
                )

            tick = time.time()
            quantizers, errors = opt_sequential(model, dataloader, DEV, args)
            print(f"Total quant + H time elapsed: {time.time() - tick:.2f}s")
            print("")
            print(
                f"Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}"
            )
            print("Quantization done.")
            print("")

    # if args.benchmark:
    #     gpus = [
    #         torch.device('cuda:%d' % i)
    #         for i in range(torch.cuda.device_count())
    #     ]
    #     if len(gpus) > 1:
    #         opt_multigpu(model, gpus)
    #     else:
    #         model = model.to(DEV)
    #     if args.benchmark:
    #         input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    #         benchmark(model, input_ids, check=args.check)
    # if args.load:
    #     exit()

    if args.save:
        #     opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

    if not args.proxy_only:
        # for dataset in ['wikitext2', 'ptb', 'c4']:
        for dataset in ["wikitext2", "ptb-new", "c4-new"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            benchmark(model, testloader, benchmark_samples=32)
