import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

#from data_reader import DataReader, Word2vecDataset
from reading_data import DataReader, Word2vecDataset
from model import SkipGramModel


class Word2VecTrainer:
    #def __init__(self, input_file, output_file, emb_dimension=128, batch_size=50, window_size=7, iterations=5,
                 #initial_lr=0.025, min_count=5, care_type=0):
    def __init__(self, args):
        self.data = DataReader(args.input_file, args.min_count, args.care_type)
        dataset = Word2vecDataset(self.data, args.window_size)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_file_name = args.output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metapath2vec")
    parser.add_argument('--input_file', type=str, help="input_file")
    parser.add_argument('--output_file', type=str, help='output_file')
    #parser.add_argument('--input_file', type=str,
                        #default="/Users/ziqiaom/Desktop/in_aminer/aminer.txt", help="input_file")
    #parser.add_argument('--output_file', type=str, default="/Users/ziqiaom/Desktop/saving/out.vec", help='output_file')
    #parser.add_argument('--input_file', type=str,
                        #default="/Users/ziqiaom/Desktop/in_dbis/dbis.txt", help="input_file")
    #parser.add_argument('--output_file', type=str, default="/Users/ziqiaom/Desktop/saving/out.vec", help='output_file')
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    parser.add_argument('--iterations', default=3, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")
    parser.add_argument('--num_workers', default=128, type=int, help="number of workers")
    args = parser.parse_args()
    #w2v = Word2VecTrainer(input_file="/Users/ziqiaom/Desktop/in_dbis/dbis.txt", output_file="/Users/ziqiaom/Desktop/saving/out.vec")
    #w2v = Word2VecTrainer(input_file=args.input_file, output_file=args.output_file)
    w2v = Word2VecTrainer(args)
    #w2v = Word2VecTrainer(input_file="/home/ubuntu/metapath2vec dgl version/venv/lib/python3.7/in_aminer/aminer.txt", output_file="/home/ubuntu/metapath2vec dgl version/venv/lib/python3.7/saving/out.vec")
    w2v.train()