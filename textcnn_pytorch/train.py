import os
import sys
import torch
import torch.nn.functional as F # 导入F函数, 用于计算损失函数


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    # L2正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        # for batch in train_iter:
        for labels, texts, offsets in train_iter:  # CHANGED: 添加offsets参数
            model.train()
            if args.cuda:
                labels, texts, offsets = labels.cuda(), texts.cuda(), offsets.cuda()  # CHANGED: 添加offsets的cuda处理
            
            optimizer.zero_grad() # 清空梯度
            logit = model(texts, offsets)   # CHANGED: 调用model时添加offsets参数
            loss = F.cross_entropy(logit, labels)
            loss.backward() # 计算梯度
            optimizer.step() # 更新参数

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()
                accuracy = 100.0 * corrects/labels.size(0)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             labels.size(0)))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for labels, texts, offsets in data_iter:  # CHANGED: 添加offsets参数
        if args.cuda:
            labels, texts, offsets = labels.cuda(), texts.cuda(), offsets.cuda()  # CHANGED: 添加offsets的cuda处理
        
        logit = model(texts, offsets)  # CHANGED: 调用model时添加offsets参数
        loss = F.cross_entropy(logit, labels, reduction='sum')

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()

    size = sum([len(batch[0]) for batch in data_iter])
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, tokenizer, vocab, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    tokens = tokenizer(text)  # CHANGED: 先分词
    text_indices = vocab(tokens)  # CHANGED: 使用vocab将分词转换为索引
    x = torch.tensor(text_indices).unsqueeze(0)  # CHANGED: 使用text_indices创建张量
    offsets = torch.tensor([0])  # CHANGED: 添加offsets参数
    if cuda_flag:
        x = x.cuda()
        offsets = offsets.cuda()  # CHANGED: 添加offsets的cuda处理
    _, output = model(x, offsets, return_features=True)  # CHANGED: 调用model时添加offsets参数
    # _, predicted = torch.max(output, 1)
    # return predicted.item() + 1  # CHANGED: 调整返回值，假设标签从1开始
    return output


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
