import torch
import os
from tqdm import tqdm
from nets.unet_training import Focal_Loss, CE_Loss, Dice_Loss
from utils_metrics import f_score
from utils import get_lr


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, 
                  num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes)
            
            if dice_loss:
                main_dice = Dice_Loss(outputs, labels)
                loss += main_dice
            
            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            from torch import amp
            with amp.autocast(device_type='cuda'):
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes)
                
                if dice_loss:
                    main_dice = Dice_Loss(outputs, labels)
                    loss += main_dice
                
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score' : total_f_score / (iteration + 1),
                'lr' : get_lr(optimizer)
            })
            pbar.update(1)
    
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1} / {Epoch}', postfix=dict, mininterval=0.3)
    

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, cls_weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, cls_weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_Loss(outputs, labels)
                loss += main_dice
            
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss' : total_loss / (iteration + 1),
                'f_score' : total_f_score / (iteration + 1),
                'lr' : get_lr(optimizer)
            })
            pbar.update(1)
    
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
        print('total_loss: %.3f || val_loss: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val))


        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch+1, total_loss/epoch_step, val_loss/epoch_step_val))))
        
        if len(loss_history) <= 1 or (val_loss/epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_epoch_weights.pth'))
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'last_epoch_weights.pth'))

    






