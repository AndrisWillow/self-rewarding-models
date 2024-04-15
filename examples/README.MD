# Some useful code snipets to run for sanity checks

Found a useful command to clone directory of a monolith repoistory:

git clone -n --depth=1 --filter=tree:0 \
  https://github.com/{User_name}/{Repository_name}
cd {Repository_name}
git sparse-checkout set --no-cone {Directory_to_clone}
git checkout

For our specific use case:
git clone -n --depth=1 --filter=tree:0 \
  https://github.com/google-research/google-research
cd google-research
git sparse-checkout set --no-cone instruction_following_eval
git checkout