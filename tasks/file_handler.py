from paips.core import Task
import os
import shutil
from pathlib import Path
import tqdm
from swissknife.aws import download_s3, S3File
import tqdm
import requests

from IPython import embed

class CopyFiles(Task):
	#Can copy from S3 but not viceversa
	def process(self):
		if not Path(self.parameters['destination_folder']).expanduser().exists():
			Path(self.parameters['destination_folder']).expanduser().mkdir(parents=True)
		copied = []
		if not isinstance(self.parameters['source_files'],list):
			self.parameters['source_files'] = [self.parameters['source_files']]
		for fname in tqdm.tqdm(self.parameters['source_files']):
			if fname.startswith('s3://'):
				fdir = fname.split('//')[1]
				file_parts = fdir.split('/')
				file_path = '/'.join(file_parts[1:])
				if self.parameters['destination_folder'].startswith('s3://'):
					download_s3(file_parts[0],file_path,'s3temp')
					destination_file = S3File(self.parameters['destination_folder'],file_path)
					destination_file.upload('s3temp')
					copied.append(str(destination_file))
				else:
					destination_file = Path(self.parameters['destination_folder'],file_path).expanduser()
					download_s3(file_parts[0],file_path,str(destination_file))
					copied.append(str(destination_file))
			else:
				fname = Path(fname).expanduser()
				if self.parameters['destination_folder'].startswith('s3://'):
					destination_file = S3File(self.parameters['destination_folder'],fname.name)
					destination_file.upload(fname)
					copied.append(str(destination_file))
				else:
					destination_file = Path(self.parameters['destination_folder'],fname.name).expanduser()
					shutil.copyfile(fname,str(destination_file))
					copied.append(str(destination_file))
		self.output_names = ['parent_folder', 'copied_files']

		return Path(self.parameters['destination_folder']).parent, copied

class Downloader(Task):
	def process(self):
		to_download = self.parameters['downloads']
		downloaded_files = []
		for download_metadata in to_download:
			destination_file = Path(download_metadata['DestinationPath']).expanduser()
			if not destination_file.parent.exists():
				destination_file.parent.mkdir(parents=True)
			if not destination_file.exists():
				if download_metadata['SourcePath'].startswith('http:/') or download_metadata['SourcePath'].startswith('https:/'):
					response = requests.get(download_metadata['SourcePath'], stream = True)
					download_file = open(str(destination_file),'wb')
					if 'Content-Length' in response.headers:
						pbar = tqdm.tqdm(unit="MB", total=int(response.headers['Content-Length'])/(1024*1024))
					else:
						pbar = None
					for chunk in tqdm.tqdm(response.iter_content(chunk_size=1024*1024)):
						if chunk:
							if pbar:
								pbar.update(len(chunk)/(1024*1024))
							download_file.write(chunk)
					download_file.close()
				elif download_metadata['SourcePath'].startswith('s3:/'):
					from swissknife.aws import download_s3
					bucket_name = download_metadata['SourcePath'].split('s3://')[-1].split('/')[0]
					bucket_path = '/'.join(download_metadata['SourcePath'].split('s3://')[-1].split('/')[1:])
					download_s3(bucket_name,bucket_path,str(destination_file))
			else:
				if self.logger:
					self.logger.info('Skipping download as {} already exists'.format(str(destination_file)))
				else:
					print('Skipping download as {} already exists'.format(str(destination_file)))
			if download_metadata.get('Extract',False) == True:
				if destination_file.suffix == '.gz':
					import tarfile
					if self.logger:
						self.logger.info('Extracting downloaded files')
					else:
						print('Extracting downloaded files')
					tar = tarfile.open(destination_file)
					tar_members = tar.getnames()
					for member in tqdm.tqdm(tar_members):
						if not Path(destination_file.parent,member).expanduser().exists():
							tar.extract(member,destination_file.parent)
					tar.close()
				elif destination_file.suffix == '.zip':
					import zipfile
					zip_file = zipfile.ZipFile(destination_file)
					zip_members = zip_file.namelist()
					for member in tqdm.tqdm(zip_members):
						if not Path(destination_file.parent,member).expanduser().exists():
							zip_file.extract(member,destination_file.parent)
					zip_file.close()
				downloaded_files.append(str(destination_file.parent))
			else:
				downloaded_files.append(str(destination_file))

		return downloaded_files

class RemoveFiles(Task):
	def process(self):
		folder_to_remove = Path(self.parameters['folder']).expanduser()
		if folder_to_remove.exists():
			shutil.rmtree(str(folder_to_remove))

		self.output_names = ['parent_folder']
		return Path(self.parameters['folder']).parent

class PathExtractor(Task):
	def process(self):
		in_paths = self.parameters['in']
		field = self.parameters['field']
		prefix = self.parameters.get('prefix','')
		suffix = self.parameters.get('suffix','')
		if prefix != '':
			prefix = prefix + '/'
		if suffix != '':
			suffix = '/' + suffix

		def extract(p,field):
			if field == 'filename':
				return Path(p).name
			elif field == 'stem':
				return Path(p).stem
			elif isinstance(field,int):
				return Path(p).parts[field]

		if isinstance(in_paths,list):
			out_paths = [prefix + extract(p,field) + suffix for p in in_paths]
			if len(out_paths) == 1:
				out_paths = out_paths[0]
			return out_paths
		else:
			return prefix + extract(in_paths,field) + suffix
			