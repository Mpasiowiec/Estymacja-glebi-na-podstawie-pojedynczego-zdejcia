# Copied from https://github.com/prs-eth/Marigold/blob/main/src/util/loss.py

import torch

def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = MSELossWithMask(**kwargs)
    elif "mae_loss" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mare_loss" == loss_name:
        criterion = MeanAbsRelLoss(**kwargs)
    elif "ge_loss" == loss_name:
        criterion = GELoss(**kwargs)
    elif "ssim_loss" == loss_name:
        criterion = SSIMLoss(**kwargs)
    elif "com_loss" == loss_name:
        criterion = combinedLoss(**kwargs)
        
        
        
        
        
        
    elif "ssitrim" == loss_name:
        criterion = SSITrim()
    elif "reg" == loss_name:
        criterion = REG()
    elif "ssitrim_reg" == loss_name:
        criterion = ssitrim_reg()
    else:
        raise NotImplementedError

    return criterion


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True, batch_reduction=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term) * self.alpha
        if self.batch_reduction:
            loss = loss.mean()        
        return loss

class MSELossWithMask:
    def __init__(self, alpha=1, batch_reduction=True):
        self.alpha = alpha
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]
        loss = self.alpha * torch.sum(torch.pow(diff,2), (-1, -2)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss

class L1LossWithMask:
    def __init__(self, alpha=1, batch_reduction=True):
        self.alpha = alpha
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]
        loss = self.alpha * torch.sum(torch.abs(diff), (-1, -2)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss

class MeanAbsRelLoss:
    def __init__(self, alpha=1, batch_reduction=True):
        self.alpha = alpha
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]
        loss = self.alpha * torch.sum(torch.abs(diff / depth_gt), (-1,-2)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss

class GELoss():
    def __init__(self, alpha=1, batch_reduction=True):
        self.alpha = alpha
        self.batch_reduction = batch_reduction
    
    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        x_grad_gt = depth_gt[:,:,:, 0:-2] - depth_gt[:,:,:, 2:]
        x_grad_pr = depth_pred[:,:,:, 0:-2] - depth_pred[:,:,:, 2:]
        y_grad_gt = depth_gt[:,:,0:-2, :] - depth_gt[:,:,2:, :]
        y_grad_pr = depth_pred[:,:,0:-2, :] - depth_pred[:,:,2:, :]
        x_grad_dif = x_grad_gt - x_grad_pr
        y_grad_dif = y_grad_gt - y_grad_pr
        x_mask = torch.mul(valid_mask[:,:,:, 0:-2], valid_mask[:,:,:, 2:])
        y_mask = torch.mul(valid_mask[:,:,0:-2, :], valid_mask[:,:,2:, :])
        if valid_mask is not None:
            x_grad_dif[~x_mask] = 0
            y_grad_dif[~y_mask] = 0
            n_x = x_mask.sum((-1, -2))
            n_y = y_mask.sum((-1, -2))
        else:
            n_x = x_grad_gt.shape[-2] * x_grad_gt.shape[-1]
            n_y = y_grad_gt.shape[-2] * y_grad_gt.shape[-1]
        loss = self.alpha * (torch.sum(torch.abs(x_grad_dif), (-1, -2)) / n_x + torch.sum(torch.abs(y_grad_dif), (-1, -2)) / n_y)
        if self.batch_reduction:
            loss = loss.mean()
        return loss

class SSIMLoss:
    def __init__(self, alpha=1, batch_reduction=True):
        self.alpha = alpha
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        gt = depth_gt.clone()
        pred = depth_pred.clone()
        if valid_mask is not None:
            gt[~valid_mask] = 0
            pred[~valid_mask] = 0
        (std_gt, mean_gt) = torch.std_mean(gt,dim=(-1,-2))
        (std_pred, mean_pred) = torch.std_mean(pred,dim=(-1,-2))
        ssim = (2*mean_pred*mean_gt + 1.0e-08)*(2*std_pred*std_gt + 1.0e-08)/((mean_pred*mean_pred + mean_gt*mean_gt + 1.0e-08)*(std_pred*std_pred + std_gt*std_gt + 1.0e-08))
        loss = self.alpha * (1 - 0.5 * ssim)
        if self.batch_reduction:
            loss = loss.mean()
        return loss

class combinedLoss:
    def __init__(self, batch_reduction=True):
        self.batch_reduction = batch_reduction


    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        
        x_grad_gt = depth_gt[:,:,:, 0:-2] - depth_gt[:,:,:, 2:]
        x_grad_pr = depth_pred[:,:,:, 0:-2] - depth_pred[:,:,:, 2:]
        y_grad_gt = depth_gt[:,:,0:-2, :] - depth_gt[:,:,2:, :]
        y_grad_pr = depth_pred[:,:,0:-2, :] - depth_pred[:,:,2:, :]
        x_grad_dif = x_grad_gt - x_grad_pr
        y_grad_dif = y_grad_gt - y_grad_pr        
        
        gt = depth_gt.clone()
        pred = depth_pred.clone()
        
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
            
            x_mask = torch.mul(valid_mask[:,:,:, 0:-2], valid_mask[:,:,:, 2:])
            y_mask = torch.mul(valid_mask[:,:,0:-2, :], valid_mask[:,:,2:, :])
            x_grad_dif[~x_mask] = 0
            y_grad_dif[~y_mask] = 0
            n_x = x_mask.sum((-1, -2))
            n_y = y_mask.sum((-1, -2))
            
            gt[~valid_mask] = 0
            pred[~valid_mask] = 0
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]
            
            n_x = x_grad_gt.shape[-2] * x_grad_gt.shape[-1]
            n_y = y_grad_gt.shape[-2] * y_grad_gt.shape[-1]
        
        loss_mae = torch.sum(torch.abs(diff), (-1, -2)) / n
        
        loss_grad = (torch.sum(torch.abs(x_grad_dif), (-1, -2)) / n_x + torch.sum(torch.abs(y_grad_dif), (-1, -2)) / n_y)
        
        (std_gt, mean_gt) = torch.std_mean(gt,dim=(-1,-2))
        (std_pred, mean_pred) = torch.std_mean(pred,dim=(-1,-2))
        ssim = (2*mean_pred*mean_gt + 1.0e-08)*(2*std_pred*std_gt + 1.0e-08)/((mean_pred*mean_pred + mean_gt*mean_gt + 1.0e-08)*(std_pred*std_pred + std_gt*std_gt + 1.0e-08))
        loss_ssim = 1 - 0.5 * ssim
        
        loss = 0.6 * loss_mae + 0.2 * loss_grad + loss_ssim
        
        if self.batch_reduction:
            loss = loss.mean()
        return loss



    
class SSITrim():
    def __init__(self, cutoff=0.2):
        self.cutoff = cutoff
        
    def __call__(self, prediction_d, gt_d, mask):

        prediction_d_a = align_depth_least_square(gt_arr=gt_d, pred_arr=prediction_d, valid_mask_arr=mask, return_scale_shift=False)
        diff = prediction_d_a - gt_d
        diff[~mask] = 0
        abs_diff = torch.abs(diff)
        sorted, _ = torch.sort(abs_diff.reshape(abs_diff.shape[0],abs_diff.shape[1],-1))
        M = mask.sum((-1,-2))
        m = (M * 0.2).int()
        for i, lim in enumerate(m):
            sorted[i][0][-lim:] = 0
        img_loss = torch.sum(sorted, dim=-1)/M
        loss = (img_loss / 2).mean()
        return loss
        
              
class REG():
    def __init__(self, scale_lv=4):
        self.scale_lv=scale_lv
    
    def single_scale_grad_loss(self, prediction_d_a, gt_d, mask):
        d_diff = prediction_d_a - gt_d
        d_diff = torch.mul(d_diff, mask)
        v_gradient = torch.abs(d_diff[:,:,0:-2, :] - d_diff[:,:,2:, :])
        v_mask = torch.mul(mask[:,:,0:-2, :], mask[:,:,2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)
        h_gradient = torch.abs(d_diff[:,:,:, 0:-2] - d_diff[:,:,:, 2:])
        h_mask = torch.mul(mask[:,:,:, 0:-2], mask[:,:,:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)
        gradient_loss = h_gradient.sum() + v_gradient.sum()
        valid_num = torch.sum(h_mask) + torch.sum(v_mask)
        gradient_loss = gradient_loss / (valid_num + 1e-8)
        return gradient_loss
    
    def __call__(self, prediction_d, gt_d, mask):

        prediction_d_a = align_depth_least_square(gt_arr=gt_d, pred_arr=prediction_d, valid_mask_arr=mask, return_scale_shift=False)
        grad_term = 0
        for i in range(self.scale_lv):
            step = pow(2,i)
            prediction_d_a = prediction_d_a[:,:,::step,::step]
            gt_d = gt_d[:,:,::step,::step]
            mask = mask[:,:,::step,::step]
            grad_term += self.single_scale_grad_loss(prediction_d_a, gt_d, mask)

        return grad_term

    
class ssitrim_reg():
    def __init__(self, alpha = 0.5, scale_lv=4):
        self.alpha = alpha
        self.scale_lv=scale_lv
        
    def single_scale_grad_loss(self, prediction_d_a, gt_d, mask):
        d_diff = prediction_d_a - gt_d
        d_diff = torch.mul(d_diff, mask)
        v_gradient = torch.abs(d_diff[:,:,0:-2, :] - d_diff[:,:,2:, :])
        v_mask = torch.mul(mask[:,:,0:-2, :], mask[:,:,2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)
        h_gradient = torch.abs(d_diff[:,:,:, 0:-2] - d_diff[:,:,:, 2:])
        h_mask = torch.mul(mask[:,:,:, 0:-2], mask[:,:,:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)
        gradient_loss = h_gradient.sum() + v_gradient.sum()
        valid_num = torch.sum(h_mask) + torch.sum(v_mask)
        gradient_loss = gradient_loss / (valid_num + 1e-8)
        return gradient_loss   
    
    def __call__(self, prediction_d, gt_d, mask):
        
        prediction_d_a = align_depth_least_square(gt_arr=gt_d, pred_arr=prediction_d, valid_mask_arr=mask, return_scale_shift=False)
        diff = prediction_d_a - gt_d
        diff[~mask] = 0
        abs_diff = torch.abs(diff)
        sorted, _ = torch.sort(abs_diff.reshape(abs_diff.shape[0],abs_diff.shape[1],-1))
        M = mask.sum((-1,-2))
        m = (M * 0.2).int()
        for i, lim in enumerate(m):
            sorted[i][0][-lim:] = 0
        img_loss = torch.sum(sorted, dim=-1)/M
        loss = (img_loss / 2).mean()

        prediction_d_a = align_depth_least_square(gt_arr=gt_d, pred_arr=prediction_d, valid_mask_arr=mask, return_scale_shift=False)
        grad_term = 0
        for i in range(self.scale_lv):
            step = pow(2,i)
            prediction_d_a = prediction_d_a[:,:,::step,::step]
            gt_d = gt_d[:,:,::step,::step]
            mask = mask[:,:,::step,::step]
            grad_term += self.single_scale_grad_loss(prediction_d_a, gt_d, mask)
        
        final_loss = loss + self.alpha * grad_term
        return final_loss