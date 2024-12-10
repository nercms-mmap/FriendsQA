# from huggingface_hub import HfApi
# from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo


# # create_repo("ZQFive/FriendsINS", token="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" ,repo_type="model")
# api = HfApi(token="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")

# api.upload_large_folder(folder_path="./shot_ins", repo_id="ZQFive/shot_ins",repo_type="model", print_report_every=1)


from huggingface_hub import Repository

# 设置你 Hugging Face Hub 上的模型/数据集名
repo_name = "ZQFive/shot_ins"  # 替换为你的用户名和仓库名
local_dir = "./shot_ins"  # 替换为本地文件夹路径

# 初始化 Repository 对象
repo = Repository(local_dir=local_dir, clone_from=repo_name, token="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")


# 添加文件到仓库并推送到 Hugging Face Hub
repo.push_to_hub(commit_message="Initial upload")