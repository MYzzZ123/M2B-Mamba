import torch

def train_model_process(model, train_loader, config, store_path_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0

        for sc, fc, labels in train_loader:
            optimizer.zero_grad()

            sc = sc.to(device)
            fc = fc.to(device)
            sc = torch.nan_to_num(sc, nan=0.0)
            fc = torch.nan_to_num(fc, nan=0.0)
            labels = labels.to(device)

            pm, ps, pf, A1, A2, A1_, A2_ = model(sc, fc, 'train')
            loss = criterion(pm, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * sc.size(0)

    torch.save(model.state_dict(), store_path_model)