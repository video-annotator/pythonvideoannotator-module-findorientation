import pyforms, math, cv2, numpy as np
from pysettings import conf
from pyforms import BaseWidget
from pyforms.Controls import ControlDir
from pyforms.Controls import ControlNumber
from pyforms.Controls import ControlList
from pyforms.Controls import ControlCombo
from pyforms.Controls import ControlSlider
from pyforms.Controls import ControlImage
from pyforms.Controls import ControlButton
from pyforms.Controls import ControlCheckBox
from pyforms.Controls import ControlCheckBoxList
from pyforms.Controls import ControlEmptyWidget
from pyforms.Controls import ControlProgress

from mcvapi.blobs.order_by_position import combinations

from pythonvideoannotator_models_gui.dialogs import DatasetsDialog
from pythonvideoannotator_models_gui.models.video.objects.object2d.datasets.contours import Contours
from pythonvideoannotator.utils.tools import points_angle, min_dist_angles


class FindOrientationWindow(BaseWidget):

	def __init__(self, parent=None):
		super(FindOrientationWindow, self).__init__('Background finder', parent_win=parent)
		self.mainwindow = parent

		if conf.PYFORMS_USE_QT5:
			self.layout().setContentsMargins(5,5,5,5)
		else:
			self.layout().setMargin(5)
		
		self.setMinimumHeight(300)
		self.setMinimumWidth(300)

		self._panel			= ControlEmptyWidget('Videos')
		self._progress  	= ControlProgress('Progress')
		self._apply 		= ControlButton('Apply', checkable=True)

		self._min_steps 	= ControlSlider('Minimum steps', 20, 1, 1000)
		self._min_dist 		= ControlSlider('Minumum distance', 30, 1, 1000)

		self._panel.value = self.contours_dialog = DatasetsDialog(self)
		self.contours_dialog.datasets_filter = lambda x: isinstance(x, Contours)

		
		self._formset = [			
			'_panel',
			('_min_steps','_min_dist'),
			'_apply',
			'_progress'
		]

		self._apply.value			= self.__apply_event
		self._apply.icon 			= conf.ANNOTATOR_ICON_PATH

		self._progress.hide()

	def init_form(self):
		super(FindOrientationWindow, self). init_form()
		
	###########################################################################
	### EVENTS ################################################################
	###########################################################################



	###########################################################################
	### PROPERTIES ############################################################
	###########################################################################
	
	
	
	def __apply_event(self):

		if self._apply.checked and self._apply.label == 'Apply':
			self._panel.enabled 		= False
			self._apply.label 			= 'Cancel'
			self._min_dist.enabled 		= False
			self._min_steps.enabled 	= False

			total_2_analyse  = 0
			for video, (begin, end), datasets in self.contours_dialog.selected_data:
				capture 		 = video.video_capture
				total_2_analyse += end-begin

			self._progress.min = 0
			self._progress.max = total_2_analyse
			self._progress.show()

			for video, (begin, end), datasets in self.contours_dialog.selected_data:
				if len(datasets)==0: continue
				begin, end = int(begin), int(end)+1
				
				back_frames  = []
				front_frames = []
				for dataset in datasets:
					for i in range(begin, end):
						vel = dataset.get_velocity(i)
						p0  = dataset.get_position(i)

						if vel is not None and p0 is not None:
							p1 = p0[0]-vel[0], p0[1]-vel[1]

							vel_angle = points_angle(p0, p1)
							angle 	  = dataset.get_angle(i)
							diff 	  = min_dist_angles(angle, vel_angle)
							
							if diff>np.pi:
								back_frames.append(i)
							else:
								front_frames.append(i)

					if len(back_frames)>len(front_frames):
						for i in back_frames:
							head, tail = dataset.get_extreme_points(i)
							centroid   = dataset.get_position(i)
							dataset.set_angle(i, points_angle(centroid, tail) )
					else:
						for i in front_frames:
							head, tail = dataset.get_extreme_points(i)
							centroid   = dataset.get_position(i)
							dataset.set_angle(i, points_angle(centroid, head) )



			self._min_dist.enabled 		= True
			self._min_steps.enabled 	= True
			self._panel.enabled 		= True
			self._apply.label 			= 'Apply'
			self._apply.checked 		= False
			self._progress.hide()



if __name__ == '__main__': 
	pyforms.startApp(TrackingWindow)
