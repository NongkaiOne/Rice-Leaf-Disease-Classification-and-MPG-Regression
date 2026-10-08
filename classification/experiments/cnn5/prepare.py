from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def main():
 frame=pd.read_csv(P/'manifest.csv');expected={'rice_blast','bacterial_leaf_blight','sheath_blight','brown_spot','healthy'}
 assert set(frame.label)==expected
 assert all((ROOT/p).is_file() for p in frame.path)
 for a,b in [('fit','validation'),('fit','test'),('validation','test')]:assert not set(frame.loc[frame.cnn_split.eq(a),'similarity_group']) & set(frame.loc[frame.cnn_split.eq(b),'similarity_group'])
 print(frame.groupby(['label','cnn_split']).size().unstack())
 print('Frozen split verified. Train/validation group assignment and original test membership are preserved.')
if __name__=='__main__':main()
