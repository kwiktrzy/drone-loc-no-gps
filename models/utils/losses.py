def get_loss(loss_name, **kwargs):
    from pytorch_metric_learning import losses
    from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=kwargs.get("temperature", 0.07))
    if loss_name == "CircleLoss":
        return losses.CircleLoss(
            m=kwargs.get("m", 0.4), gamma=kwargs.get("gamma", 80)
        )
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=kwargs.get("alpha", 1.0),
            beta=kwargs.get("beta", 50),
            base=kwargs.get("base", 0.0),
            distance=kwargs.get("distance", CosineSimilarity())
        )
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(
            pos_margin=kwargs.get("pos_margin", kwargs.get("margin", 0.8)),
            neg_margin=kwargs.get("neg_margin", 0.5),
            distance=kwargs.get("distance", CosineSimilarity())
        )
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=kwargs.get("neg_margin", 0),
            pos_margin=kwargs.get("pos_margin", 1),
            distance=kwargs.get("distance", DotProductSimilarity())
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=kwargs.get("num_bins", 30))
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(temperature=kwargs.get("temperature", 0.07))
    if loss_name == "TripletMarginLoss":
        return losses.TripletMarginLoss(
            margin=kwargs.get("margin", 0.03),
            swap=kwargs.get("swap", False),
            smooth_loss=kwargs.get("smooth_loss", False),
            distance=kwargs.get("distance", CosineSimilarity()),
        )
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=kwargs.get("margin", 0.05),
            swap=kwargs.get("swap", False),
            smooth_loss=kwargs.get("smooth_loss", False),
            triplets_per_anchor=kwargs.get("triplets_per_anchor", "all"),
        )
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


def get_miner(miner_name, **kwargs):
    from pytorch_metric_learning import miners
    from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

    if miner_name == "TripletMarginMiner":
        return miners.TripletMarginMiner(
            margin=kwargs.get("margin", 0.1),
            type_of_triplets=kwargs.get("type_of_triplets", "all"),
            distance=kwargs.get("distance", CosineSimilarity()),
        )
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(
            epsilon=kwargs.get("epsilon", kwargs.get("margin", 0.1)),
            distance=kwargs.get("distance", CosineSimilarity())
        )
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=kwargs.get("pos_margin", 0.8),
            neg_margin=kwargs.get("neg_margin", 0.6),
            distance=kwargs.get("distance", DotProductSimilarity())
        )
    return None