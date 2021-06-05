import pandas as pd

if __name__  == "__main__":
    datafile = "/vinai/haopt12/meanteacher/logs/cifar100_resnet26_test.csv"
    df = pd.read_csv(datafile, delim_whitespace=True, header=None)
    df.columns = [
        "student_sup_loss",
        "teacher_sup_loss",
        "student_top1_acc",
        "student_top5_acc",
        "student_top1_error",
        "student_top5_error",
        "teacher_top1_acc",
        "teacher_top5_acc",
        "teacher_top1_error",
        "teacher_top5_error"]
    desc = df.describe()
    print(desc)
    desc.to_csv("logs/desc.csv")
    

    