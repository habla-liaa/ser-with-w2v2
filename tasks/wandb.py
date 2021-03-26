from paips.core import Task
import glob
from pathlib import Path
import wandb
from paips.utils import find_cache, GenericFile
import joblib

class WANDBExperiment(Task):
    def clear_cache(self):
        import time
        import os
        import shutil

        cache_path = Path('~/.cache/wandb').expanduser()

        while True:
            if cache_path.exists():
                shutil.rmtree(cache_path)

            time.sleep(60)

    def process(self):
        project_name = self.parameters.get('project_name',None)
        clear_cache = self.parameters.get('clear_cache', True)
        #if not project_name:
        #    raise Exception('Please provide a project name for this experiment')
        experiment_name = self.parameters.get('experiment_name',None)
        description = self.parameters.get('description',None)

        wandb_id = wandb.util.generate_id()
        run_obj = wandb.init(id = wandb_id, name=experiment_name, project=project_name, notes=description, config=self.config, resume='allow')

        if clear_cache:
            #Starts daemon thread to clear regularly cache folder
            import threading
            gc_thread = threading.Thread(target=self.clear_cache,daemon=True)
            gc_thread.start()

        return wandb_id

    def find_cache(self):
        cache_paths = find_cache(self.task_hash,self.global_parameters['cache_path'])
        if cache_paths and self.cache:
            wdb_file = GenericFile(cache_paths[0])
            if Path(wdb_file.local_filename).exists():
                wandb_id = joblib.load(Path(wdb_file.local_filename))
            elif wdb_file.exists():
                wdb_file.download()
                wandb_id = joblib.load(Path(wdb_file.local_filename))

            project_name = self.parameters.get('project_name',None)
            experiment_name = self.parameters.get('experiment_name',None)
            description = self.parameters.get('description',None)

            wandb.init(id = wandb_id, name=experiment_name, project=project_name, notes=description, config=self.config, resume=wandb_id)

        return cache_paths