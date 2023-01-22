import sys
import pandas as pd
import numpy as np

def main():
    # total arguments
    n = len(sys.argv)
    # print("Total arguments passed:", n)
    if(n!=5):
        print("Number of arguements is invalid")
        exit(0)

    # Arguments passed
    # print("\nName of Python script:", sys.argv[0])
    try:
        df=pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print("The file you're trying to access does not exist")
        raise

    l=len(df.columns)
    try:
        if(l<3):
            raise
    except:
        print("Number of columns are ",l,". It should be more than 3.")
        exit(0)

    print(df)
    new_df=df.iloc[:,1:]
    # print(new_df)

    # print(sys.argv[2])
    # print(type(sys.argv[2]))
    weight=np.array(sys.argv[2].split(","))
    # print(weight,type(weight))

    impact=np.array(sys.argv[3].split(","))
    # print(len(weight))
    # print(impact,type(impact))
    if(np.size(weight)!=np.size(impact)):
        print("Inapprpriate Dimensions1")
        exit(0)
    elif(np.size(weight)!=l-1):
        print("Inapprpriate Dimensions2")
        exit(0)

    resultFileName=sys.argv[4]
    # print(resultFileName)
    # print(type(impact[0]))
    for i in impact:
        if(i!='+' and i!='-'):
            # print(i,type(i))
            print("Imapct can be either + or - only")
            exit(0)
            
    nrow=len(new_df)
    sumGRE=0
    # print(new_df)
    for i in range(0,l-1):
        sumGRE=0
        dfx=new_df.iloc[:,i].values
        dfx_sq=dfx**2
        sumGRE=sum(dfx_sq)
        dfx=dfx/(sumGRE**(1/2))
        new_df.iloc[:,i]=dfx

    # print(new_df)
    # print(l)

    for i in range(0,l-1):
        edit=new_df.iloc[:,i]
        edit=edit*int(weight[i])
        new_df.iloc[:,i]=edit

    # print(new_df)

    for i in range(0,l-1):
        edit=new_df.iloc[:,i].values
        new_df.loc[nrow,new_df.columns[i]]=max(edit)
        new_df.loc[nrow+1,new_df.columns[i]]=min(edit)

    # print(new_df)
    for i in range(0,l-1):
        if(impact[i]=='-'):
            new_df.iloc[nrow,i],new_df.iloc[nrow+1,i]=new_df.iloc[nrow+1,i],new_df.iloc[nrow,i]

    # print(new_df)
    # print(new_df.iloc[:-2,:])

    SPlus=np.array(0)
    SMinus=np.array(0)
    for i in range(0,l-1):
        edit=new_df.iloc[:-2,i].values
        edit1=edit-new_df.iloc[nrow,i]
        edit2=edit-new_df.iloc[nrow+1,i]
        # print(type(edit1))
        edit1=edit1**2
        edit2=edit2**2
        SPlus=SPlus+edit1
        SMinus=SMinus+edit2

    SPlus=SPlus**0.5
    SMinus=SMinus**0.5
    # SPlus=pd.DataFrame(SPlus)
    # SMinus=pd.DataFrame(SMinus)
    # print(SPlus)
    # print(SMinus)

    # print(l)
    P=[0]*len(SPlus)
    for i in range(0,nrow):
        P[i]=SMinus[i]/(SPlus[i]+SMinus[i])



    # print(P)
    maxi=0
    maxval=P[0]
    for i in range(1,len(P)):
        if(P[i]>maxval):
            maxi=i
            maxval=P[i]

    P=pd.DataFrame(P)

    # # P.columns[0]="PerformanceScore"
    # P.rename(columns={0:"PerformanceScore"},inplace=True)
    # print(P)
    # # print(maxi,maxval)
    # # print("The best according to topsis is",df.iloc[maxi,0])
    index=pd.DataFrame(np.array(range(1,nrow+1)))
    rank=pd.DataFrame(np.array(range(1,nrow+1)))
    # print(rank)
    # P=(P.sort_values(by=['PerformanceScore'],ascending=False))
    # print(P)
    # P.loc[:,"Rank"]=rank
    # print(P)

    # print(index)


    # P=P.sort_values(by=['PerformanceScore'],ascending=False)
    df.loc[:,"Index"]=index
    df.loc[:,"PerformanceScore"]=P
    df=df.sort_values(by=['PerformanceScore'],ascending=False)
    # df["Rank"]=''
    rank=[]
    # df=pd.DataFrame(np.array(df))
    for i in range(0,nrow):
        rank.append(i+1)
    df['Rank']=rank
    df=df.sort_index()
    df.to_csv(sys.argv[4])
    print(df)

if __name__ == "__main__":
    main()