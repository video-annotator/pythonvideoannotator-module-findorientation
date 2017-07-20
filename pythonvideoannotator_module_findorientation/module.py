import cv2
from pysettings import conf
from pythonvideoannotator_module_findorientation.findorientation_window import FindOrientationWindow


class Module(object):

	def __init__(self):
		"""
		This implements the Path edition functionality
		"""
		super(Module, self).__init__()


		self.findorientation_window = FindOrientationWindow(self)


		self.mainmenu[1]['Modules'].append(
			{'Estimate the contours orientation': self.findorientation_window.show, 'icon':conf.ANNOTATOR_ICON_BACKGROUND },			
		)



	
	######################################################################################
	#### IO FUNCTIONS ####################################################################
	######################################################################################

	
	def save(self, data, project_path=None):
		data = super(Module, self).save(data, project_path)
		data['findorientation-settings'] = self.findorientation_window.save_form({})
		return data

	def load(self, data, project_path=None):
		super(Module, self).load(data, project_path)
		if 'findorientation-settings' in data: self.findorientation_window.load_form(data['findorientation-settings'])
		