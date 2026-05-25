import os
import torch
import torch.optim as optim
from config import args
from data_pipeline.loader import FusionData
from training.losses import FusionLoss
from training.utils import (
    count_parameters, setup_runtime, is_real_data, data_label, tensor_channels,
    update_learning_rate, compute_and_print_metrics,
    save_outputs, print_loss_components, print_final_results,
    print_data_info, load_mask, load_psf_srf,
    report_mask_state, create_model, train_psf_model, prepare_psf_for_main_training
)
def main():
    setup_runtime(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    is_real = is_real_data(args)
    data_type = data_label(is_real)
    print(f"Run | device {device} | seed {args.seed} | dataset {args.data_name}")
    data = FusionData(args)
    if not hasattr(data, 'tensor_hr_msi') or not hasattr(data, 'tensor_lr_hsi') or not hasattr(data, 'tensor_gt'):
        raise ValueError("Data loading failed")
    Z = data.tensor_hr_msi.to(device)
    Y = data.tensor_lr_hsi.to(device)
    gt = data.tensor_gt.to(device)
    if is_real:
        print_data_info(args.data_name, is_real, Z.shape, Y.shape, "None (Real data without GT)")
    else:
        print_data_info(args.data_name, is_real, Z.shape, Y.shape, gt.shape)
    mask = load_mask(args, Y.shape, device)
    Y_masked = Y.clone()
    report_mask_state(mask, Y)
    train_psf_model(args, data)
    prepare_psf_for_main_training(args)
    psf, srf = load_psf_srf(args, device)
    in_channels_msi, in_channels_hsi, out_channels, num_endmembers = tensor_channels(Z, Y, gt, args)
    print(f"Channels | MSI {in_channels_msi} | HSI {in_channels_hsi} | output {out_channels}")
    model = create_model(args, in_channels_msi, in_channels_hsi, out_channels, num_endmembers, device)
    criterion = FusionLoss(
        lambda_rec=args.lambda_rec,
        lambda_ASC=args.lambda_ASC,
        lambda_joint=args.lambda_joint,
        lambda_SPA=args.lambda_SPA,
        lambda_SPE=args.lambda_SPE,
        args=args
    ).to(device)
    params_to_optimize = list(model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.lr_stage1)
    print(f"Optimizer | parameter groups {len(params_to_optimize)}")
    psf_note = " | pre-trained PSF" if getattr(args, 'use_moesr_psf', 'No') == 'Yes' else ""
    print(f"Stage 2 | main training{psf_note}")
    save_dir = os.path.join(args.expr_dir, 'training_results')
    os.makedirs(save_dir, exist_ok=True)
    total_epochs = args.niter1 + args.niter_decay1
    if total_epochs > 0:
        print(f"Training | epochs {total_epochs} | params {count_parameters(model) / 1e6:.2f}M")
    for epoch in range(1, total_epochs + 1):
        model.train()
        should_log_detail = epoch == 1 or epoch % args.log_interval == 0 or epoch == total_epochs
        should_eval = epoch == 1 or epoch % args.eval_interval == 0 or epoch == total_epochs
        should_save_outputs = epoch == 1 or epoch % args.output_interval == 0 or epoch == total_epochs
        should_checkpoint = epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == total_epochs
        optimizer.zero_grad(set_to_none=True)
        Z_hat, Y_hat, X_hat, A, A_tilde = model(Z, Y_masked, mask)
        ratio = args.scale_factor
        loss = criterion(Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask)
        loss.backward()
        optimizer.step()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning | epoch {epoch} | abnormal loss {loss.item()}")
        if should_log_detail and not should_eval:
            loss_comps = criterion.last_loss_components or criterion.get_loss_components(
                Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask
            )
            print(
                f"Epoch {epoch}/{total_epochs} | train loss {loss.item():.6f} | "
                f"rec {loss_comps['rec']:.6f} | asc {loss_comps['asc']:.6f} | "
                f"spa {loss_comps['spa']:.6f} | spe {loss_comps['spe']:.6f}"
            )
        if epoch % 10 == 0 and not should_log_detail:
            print(f"Epoch {epoch}/{total_epochs} | loss {loss.item():.6f}")
        if should_eval:
            model.eval()
            with torch.inference_mode():
                Z_hat_eval, Y_hat_eval, X_hat_eval, A_eval, A_tilde_eval = model(Z, Y_masked, mask)
                eval_loss = criterion(Z, Y, Z_hat_eval, Y_hat_eval, X_hat_eval, A_eval, A_tilde_eval, psf, srf, ratio,
                                      mask)
                loss_comps_eval = criterion.last_loss_components or criterion.get_loss_components(
                    Z, Y, Z_hat_eval, Y_hat_eval, X_hat_eval, A_eval, A_tilde_eval, psf, srf, ratio, mask
                )
                print_loss_components(epoch, eval_loss.item(), loss_comps_eval, "Evaluation")
                if not is_real:
                    gt_np_eval = gt.cpu().numpy()[0].transpose(1, 2, 0)
                    pred_np_eval = X_hat_eval.cpu().numpy()[0].transpose(1, 2, 0)
                    compute_and_print_metrics(gt_np_eval, pred_np_eval, args.scale_factor, f"{epoch}_eval", data_type)
                if should_save_outputs:
                    save_outputs(epoch, Z_hat_eval, Y_hat_eval, X_hat_eval, A_eval, A_tilde_eval, save_dir)
            model.train()
        if should_checkpoint:
            model_path = os.path.join(args.expr_dir, 'ramoe_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint saved | {model_path}")
        update_learning_rate(optimizer, args, epoch)
        if device.type == 'cuda' and args.empty_cache_interval > 0 and epoch % args.empty_cache_interval == 0:
            torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        Z_hat_final, Y_hat_final, X_hat_final, A_final, A_tilde_final = model(Z, Y_masked, mask)
        final_loss_comps = criterion.get_loss_components(Z, Y, Z_hat_final, Y_hat_final, X_hat_final, A_final,
                                                         A_tilde_final, psf, srf, ratio, mask)
        print_final_results(final_loss_comps, save_dir)
        if not is_real:
            gt_np_final = gt.cpu().numpy()[0].transpose(1, 2, 0)
            pred_np_final = X_hat_final.cpu().numpy()[0].transpose(1, 2, 0)
            compute_and_print_metrics(gt_np_final, pred_np_final, args.scale_factor, "final", data_type)
        save_outputs('final', Z_hat_final, Y_hat_final, X_hat_final, A_final, A_tilde_final, save_dir)
if __name__ == '__main__':
    main()
