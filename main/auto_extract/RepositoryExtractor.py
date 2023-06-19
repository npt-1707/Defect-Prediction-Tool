import os
import numpy as np
import pandas as pd
from github import Github
from tqdm import tqdm
from auto_extract.utils.utils import *
import pickle
import datetime


class RepositoryExtractor:
    def __init__(self, repo_path: str, save_path: str, language:str):
        self.repo_path = repo_path
        name = os.path.basename(os.path.normpath(repo_path))
        self.save_path = os.path.join(save_path, "auto_extract", "save", name)
        
        self.commits_path = os.path.join(
            self.save_path, f"commits.pkl")
        self.last_hash_path = os.path.join(
            self.save_path, f"last_hash.pkl")
        self.features_path = os.path.join(
            self.save_path, f"features.pkl")
        self.files_path = os.path.join(
            self.save_path, f"files.pkl")
        self.authors_path = os.path.join(
            self.save_path, f"authors.pkl")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # get commit hashes
        os.chdir(self.repo_path)
        # test updating function of info
        # date = "2023-01-15" 
        # date = "2023-02-20"
        # date = "2023-06-01"
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.commit_ids = get_commit_hashes(date)[::-1]
        
        # load the existing info
        self.commits = {}
        self.files = {}
        self.authors = {}
        self.features = {}
        self.last_hash = [None, None]

        if os.path.exists(self.commits_path):
            with open(self.commits_path, "rb") as f:
                self.commits = pickle.load(f)

        if os.path.exists(self.files_path):
            with open(self.files_path, "rb") as f:
                self.files = pickle.load(f)

        if os.path.exists(self.authors_path):
            with open(self.authors_path, "rb") as f:
                self.authors = pickle.load(f)
                
        if os.path.exists(self.features_path):
            with open(self.features_path, "rb") as f:
                self.features = pickle.load(f)
                
        if os.path.exists(self.last_hash_path):
            with open(self.last_hash_path, "rb") as f:
                self.last_hash = pickle.load(f)
        
        self.language = language

    def get_commit_info(self, commit_id, languages=[]):
        command = "git show {} --name-only --pretty=format:'%H%n%P%n%an%n%ct%n%s%n%B%n[ALL CHANGE FILES]'"
        show_msg = exec_cmd(command.format(commit_id))
        show_msg = [msg.strip() for msg in show_msg]
        file_index = show_msg.index("[ALL CHANGE FILES]")

        head = show_msg[:5]
        commit_msg = show_msg[5:file_index]

        parent_id = head[1]
        author = head[2]
        commit_date = head[3]
        commit_msg = " ".join(commit_msg)

        command = "git show {} --pretty=format: --unified=999999999"
        diff_log = split_diff_log(exec_cmd(command.format(commit_id)))
        commit_diff = {}
        commit_blame = {}
        files = []
        for log in diff_log:
            files_diff = aggregator(parse_lines(log))
            for file_diff in files_diff:
                file_name_a = (
                    file_diff["from"]["file"]
                    if file_diff["rename"] or file_diff["from"]["mode"] != "0000000"
                    else file_diff["to"]["file"]
                )
                file_name_b = (
                    file_diff["to"]["file"]
                    if file_diff["rename"] or file_diff["to"]["mode"] != "0000000"
                    else file_diff["from"]["file"]
                )
                if file_diff["is_binary"] or len(file_diff["content"]) == 0:
                    continue

                if file_diff["from"]["mode"] == "000000000":
                    continue

                file_language = get_programming_language(file_name_b)
                if file_language is None:
                    continue
                if len(languages) > 0:
                    if file_language not in languages:
                        continue
                    
                command = "git blame -t -n -l {} '{}'"
                file_blame_log = exec_cmd(command.format(parent_id, file_name_a))
                if not file_blame_log:
                    continue
                file_blame = get_file_blame(file_blame_log)
                commit_blame[file_name_b] = file_blame
                commit_diff[file_name_b] = file_diff
                files.append(file_name_b)

        commit = {
            "commit_id": commit_id,
            "parent_id": parent_id,
            "commit_msg": commit_msg,
            "author": author,
            "commit_date": int(commit_date),
            "files": files,
            "diff": commit_diff,
            "blame": commit_blame,
        }
        return commit

    def get_repo_commits_info(self, main_language_only=False):
        if self.last_hash[0] == self.commit_ids[-1]:
            print(self.last_hash, self.commit_ids[-1])
            print("skip 1")
            return
        if main_language_only:
            languages = [self.language]
        else:
            languages = []
        print("Collecting commits information ...")
        if self.last_hash[0] is not None:
            start = self.commit_ids.index(self.last_hash[0])
        else:
            start = -1
        for idx in tqdm(range(start+1, len(self.commit_ids))):
            commit_id = self.commit_ids[idx]
            if commit_id not in self.commits:
                commit = self.get_commit_info(commit_id, languages)
                if not commit["diff"]:
                    continue
                self.commits[commit_id] = commit
                
        self.last_hash[0] = self.commit_ids[-1]
        with open(self.last_hash_path, "wb") as f:
            pickle.dump(self.last_hash, f)
            
        with open(self.commits_path, "wb") as f:
            pickle.dump(self.commits, f)

    def extract_k_features(self, commit_id):
        commit = self.commits[commit_id]
        commit_date = commit["commit_date"]
        commit_message = commit["commit_msg"]
        commit_author = commit["author"]
        commit_diff = commit["diff"]
        commit_blame = commit["blame"]

        la, ld, lt, age, nuc = (0, 0, 0, 0, 0)
        subs, dirs, files = [], [], []
        totalLOCModified = 0
        locModifiedPerFile = []
        authors = []
        ages = []
        author_exp = self.authors.get(commit_author, {})

        for file_elem in list(commit_diff.items()):
            file_path = file_elem[0]
            val = file_elem[1]

            subsystem, directory, filename = get_subs_dire_name(file_path)
            if subsystem not in subs:
                subs.append(subsystem)
            if directory not in dirs:
                dirs.append(directory)
            if filename not in files:
                files.append(filename)

            result = calu_modified_lines(val)
            la += result[0]
            ld += result[0]
            lt += result[1]

            totalLOCModified += la + ld
            locModifiedPerFile.append(totalLOCModified)

            file = self.files.get(file_path, {"author": [], "nuc": 0})
            file_author = file["author"]
            if commit_author not in file_author:
                file_author.append(commit_author)
            authors = list(set(authors) | set(file_author))

            prev_time = get_prev_time(commit_blame, file_path)
            age = commit_date - prev_time if prev_time else 0
            age = max(age, 0)
            ages.append(age)

            file_nuc = file["nuc"] + 1
            nuc += file_nuc

            file["nuc"] = file_nuc
            self.files[file_path] = file

            if file_path in author_exp:
                author_exp[file_path].append(commit_date)
            else:
                author_exp[file_path] = [commit_date]
            self.authors[commit_author] = author_exp

        feature = {
            "_id": commit_id,
            "date": commit_date,
            "ns": len(subs),
            "nd": len(dirs),
            "nf": len(files),
            "entrophy": calc_entrophy(totalLOCModified, locModifiedPerFile),
            "la": la,
            "ld": ld,
            "lt": lt,
            "fix": check_fix(commit_message),
            "ndev": len(authors),
            "age": np.mean(ages) / 86400 if ages else 0,
            "nuc": nuc,
            "exp": get_author_exp(author_exp),
            "rexp": get_author_rexp(author_exp, commit_date),
            "sexp": get_author_sexp(author_exp, subs),
        }
        return feature

    def extract_repo_k_features(self):
        if self.last_hash[1] == self.last_hash[0] or self.last_hash[1] == self.commit_ids[-1]:
            print("skip 2")
            return
        print("Extracting features ...")
        if self.last_hash[1] is not None:
            start = self.commit_ids.index(self.last_hash[1])
        else:
            start = -1
        for idx in tqdm(range(start+1, len(self.commit_ids))):
            commit_id = self.commit_ids[idx]
            if commit_id in self.commits and commit_id not in self.features:
                k_features = self.extract_k_features(commit_id)
                self.features[commit_id] = k_features
        
        self.last_hash[1] = self.commit_ids[-1]
        
        with open(self.last_hash_path, "wb") as f:
            pickle.dump(self.last_hash, f)
            
        with open(self.features_path, "wb") as f:
            pickle.dump(self.features, f)
        
        with open(self.files_path, "wb") as f:
            pickle.dump(self.files, f)

        with open(self.authors_path, "wb") as f:
            pickle.dump(self.authors, f)