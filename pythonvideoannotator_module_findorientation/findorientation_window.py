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
from pythonvideoannotator_models.utils.tools import points_angle, min_dist_angles, lin_dist

from pythonvideoannotator_models.utils.tools import savitzky_golay

from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC



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

        self._panel         = ControlEmptyWidget('Videos')
        self._progress      = ControlProgress('Progress')
        self._apply         = ControlButton('Apply', checkable=True)
        self._debug         = ControlCheckBox('Create all the intermediam values')

        self._min_steps     = ControlSlider('Minimum steps', default=20,  minimum=1, maximum=1000)
        self._min_dist      = ControlSlider('Minumum distance', default=30,  minimum=1, maximum=1000)

        self._panel.value = self.contours_dialog = DatasetsDialog(self)
        self.contours_dialog.datasets_filter = lambda x: isinstance(x, Contours)

        
        self._formset = [         
            '_panel',
            ('_min_steps','_min_dist'),
            '_debug',
            '_apply',
            '_progress'
        ]

        self._apply.value           = self.__apply_event
        self._apply.icon            = conf.ANNOTATOR_ICON_PATH

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
            self._panel.enabled         = False
            self._apply.label           = 'Cancel'
            self._min_dist.enabled      = False
            self._min_steps.enabled     = False

            total_2_analyse  = 0
            for video, (begin, end), datasets in self.contours_dialog.selected_data:
                capture          = video.video_capture
                total_2_analyse += (end-begin)*len(datasets)*5

            self._progress.min = 0
            self._progress.max = total_2_analyse
            self._progress.show()

            debug_mode = self._debug.value

            count = 0
            for video, (begin, end), datasets in self.contours_dialog.selected_data:
                if len(datasets)==0: continue
                begin, end = int(begin), int(end)
                
                back_frames  = []
                front_frames = []
                for dataset in datasets:

                    window      = 30
                    min_walked  = 50
                    savitzky_golay_window_size = (window+1 if (window % 2 )==0 else window)


                    if debug_mode:
                        # create the values for debug ##########################################
                        v1 = dataset.object2d.create_value()
                        v1.name = 'Est. orient. - step1 - smoothed walked distance (prev {0} frames)'.format(window)
                        v2 = dataset.object2d.create_value()
                        v2.name = 'Est. orient. - step1 - smoothed walked distance with direction (prev {0} frames)'.format(window)    

                        vflipped = dataset.object2d.create_value()
                        vflipped.name = 'Est. orient. - step1 - flipped images'
                        ########################################################################

                    
                    _, walked_distance     = dataset.calc_walked_distance(window)
                    _, dir_walked_distance = dataset.calc_walked_distance_with_direction(window)
                    walked_distance        = savitzky_golay(np.array(walked_distance), window_size=savitzky_golay_window_size)
                    dir_walked_distance    = savitzky_golay(np.array(dir_walked_distance), window_size=savitzky_golay_window_size)
                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                    for i in range(begin, end):
                        self._progress.value = count; count+=1
                        if i>=len(walked_distance): continue

                        vflipped.set_value(i, 0)

                        ### Debug ##############################################################
                        if debug_mode: 
                            v1.set_value(i, walked_distance[i])
                            v2.set_value(i, dir_walked_distance[i])
                        ########################################################################

                        if walked_distance[i]>min_walked and dir_walked_distance[i]<-min_walked:
                            dataset.flip(i)

                            if debug_mode: vflipped.set_value(i, 1) 

                      
                    if debug_mode:
                        # create the values for debug ##########################################
                        v1 = dataset.object2d.create_value()
                        v1.name = 'Est. orient. - step2 - smoothed walked distance (prev {0} frames)'.format(window)
                        v2 = dataset.object2d.create_value()
                        v2.name = 'Est. orient. - step2 - smoothed walked distance with direction (prev {0} frames)'.format(window)    

                        vtrainning = dataset.object2d.create_value()
                        vtrainning.name = 'Est. orient. - step2 - good images for trainning'
                        ########################################################################

                    _, walked_distance     = dataset.calc_walked_distance(window)
                    _, dir_walked_distance = dataset.calc_walked_distance_with_direction(window)
                    
                    trainning_frames = []

                    IMAGE_SIZE = 250
                    MAX_TRAINNING_SET = 200

                    model = cv2.ml.ANN_MLP_create()
                    layer_sizes = np.int32([IMAGE_SIZE**2, IMAGE_SIZE, IMAGE_SIZE, 2])

                    model.setLayerSizes(layer_sizes)
                    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
                    model.setBackpropMomentumScale(0.0)
                    model.setBackpropWeightScale(0.001)
                    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
                    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)


                    train   = []
                    answers = []
                    for i in range(begin, end):
                        self._progress.value = count; count+=1
                        if i>=len(walked_distance): continue

                        trainning_frames.append(False)
                        if debug_mode: vtrainning.set_value(i, 0)

                        ### Debug ##############################################################
                        if debug_mode: 
                            v1.set_value(i, walked_distance[i])
                            v2.set_value(i, dir_walked_distance[i])
                        ########################################################################
                        triangle_angle = dataset.get_minimumenclosingtriangle_angle(i)
                        if triangle_angle is None: continue

                        dataset_angle  = dataset.get_angle(i)
                        if dataset_angle is None:  continue

                        if  walked_distance[i]>min_walked and dir_walked_distance[i]>min_walked and \
                            min_dist_angles(dataset_angle, triangle_angle)<(np.pi/2):
                            trainning_frames[i] = True

                            if len(train)<MAX_TRAINNING_SET:

                                ok1, img_up   = dataset.get_image(i, mask=True, up=True, margin=20, size=(IMAGE_SIZE,IMAGE_SIZE) )
                                ok2, img_down = dataset.get_image(i, mask=True, angle=dataset_angle+np.pi, up=True, margin=20, size=(IMAGE_SIZE,IMAGE_SIZE) )
                                
                                if ok1 and ok2:
                                    train.append(img_up[:,:,0].flatten())
                                    train.append(img_down[:,:,0].flatten())
                                    answers.append([1,0])
                                    answers.append([0,1])

                                    #cv2.imwrite('/home/ricardo/Downloads/test/{0}.png'.format(i), img_down)
                                    if debug_mode: vtrainning.set_value(i, 1)    
                                

                        

                    train   = np.float32(train)
                    answers = np.float32(answers)
                    model.train(train, cv2.ml.ROW_SAMPLE, answers)


                    if debug_mode:
                        # create the values for debug ##########################################
                        v1 = dataset.object2d.create_value()
                        v1.name = 'Est. orient. - step3 - prediction front'
                        v2 = dataset.object2d.create_value()
                        v2.name = 'Est. orient. - step3 - prediction back'
                        v3 = dataset.object2d.create_value()
                        v3.name = 'Est. orient. - step3 - binary prediction'
                    
                    for i in range(begin, end):
                        self._progress.value = count; count+=1
                        if i>=len(trainning_frames): continue
                        
                        if debug_mode: v1.set_value(i,0)

                        if not trainning_frames[i]:

                            ok, img = dataset.get_image(i, mask=True, up=True, margin=20, size=(IMAGE_SIZE,IMAGE_SIZE) )
                            
                            if ok:
                                _ret, resp = model.predict(np.float32([img[:,:,0].flatten()]))
                                if resp.argmax()==1: dataset.flip(i) 

                                if debug_mode: 
                                    v1.set_value(i,resp[0][0])
                                    v2.set_value(i,resp[0][1]) 
                                    v3.set_value(i,resp.argmax()) 

                       

                    angles2smooth = []
                    for i in range(begin, end):
                        a = dataset.get_angle_diff_to_zero(i)

                        angles2smooth.append( 0 if a is None else a )
                        self._progress.value = count; count+=1
                    smoothedangles = savitzky_golay(np.array(angles2smooth), window_size=31) 

                    if debug_mode:
                        # create the values for debug ##########################################
                        v1 = dataset.object2d.create_value()
                        v1.name = 'Est. orient. - step4 - angles'
                        v2 = dataset.object2d.create_value()
                        v2.name = 'Est. orient. - step4 - smooth angles'
                        
                        for i in range(begin, end):
                            if i>=len(angles2smooth): break
                            if i>=len(smoothedangles): break
                            v1.set_value(i, angles2smooth[i])
                            v2.set_value(i, smoothedangles[i])

                        v1 = dataset.object2d.create_value()
                        v1.name = 'Est. orient. - step5 - flipped'

                    for i in range(begin, end):
                        self._progress.value = count; count+=1
                        angle    = dataset.get_angle_diff_to_zero(i)
                        if angle is None: continue
                        if i>=len(smoothedangles): continue
                        
                        if debug_mode: v1.set_value(i, 0)

                        smoothed = smoothedangles[i]
                        if min_dist_angles(angle, smoothed)>(np.pi/2):
                            dataset.flip(i)
                            if debug_mode: v1.set_value(i, 1)
                        
                    

                        

                        

                self._progress.value = count
                count += 1

            self._min_dist.enabled      = True
            self._min_steps.enabled     = True
            self._panel.enabled         = True
            self._apply.label           = 'Apply'
            self._apply.checked         = False
            self._progress.hide()



if __name__ == '__main__': 
    pyforms.start_app(FindOrientationWindow)
