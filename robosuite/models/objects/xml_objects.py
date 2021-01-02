from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, string_to_array


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle.xml"))


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can.xml"))


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/lemon.xml"))


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk.xml"))


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal.xml"))


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/square-nut.xml"))


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round-nut.xml"))


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in SawyerPickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk-visual.xml"))


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread-visual.xml"))


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"))


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can-visual.xml"))


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in BaxterPegInHole)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"))

class DoorPullNoLatchObject(MujocoXMLObject):
  """
  Door: pull with no latch
  """

  def __init__(self):
        #super().__init__(xml_path_completion("objects/door_dapg.xml"))  
        super().__init__(xml_path_completion("objects/door_pull_no_latch.xml"))  

  def set_goal_xpos(self, x_delta, y_delta):
      """ Sets x,y position of goal site in door model with x and y offset from door center"""

      door_center_site = self.worldbody.find("./body/body/body/site[@name='door_center']")
      door_center_pos = string_to_array(door_center_site.get("pos"))
      goal_site = self.worldbody.find("./body/body/body/site[@name='goal']")
      goal_site.set("pos", array_to_string([door_center_pos[0] + x_delta, door_center_pos[1] + y_delta, -1.0]))

  @property
  def handle_contact_geoms(self):
      return[
        "handle_base",
        "handle",
      ]

  @property
  def door_contact_geoms(self):
      return[
        "door_box",
        "door_r_cyl",
        "door_l_cyl",
        "l_frame",
        "r_frame",
      ]

class DoorPullWithLatchObject(MujocoXMLObject):
  """
  Door: pull with latch
  """

  def __init__(self):
        #super().__init__(xml_path_completion("objects/door_dapg.xml"))  
        super().__init__(xml_path_completion("objects/door_pull_with_latch.xml"))  

  def set_goal_xpos(self, x_delta, y_delta):
      """ Sets x,y position of goal site in door model with x and y offset from door center"""

      door_center_site = self.worldbody.find("./body/body/body/site[@name='door_center']")
      door_center_pos = string_to_array(door_center_site.get("pos"))
      goal_site = self.worldbody.find("./body/body/body/site[@name='goal']")
      goal_site.set("pos", array_to_string([door_center_pos[0] + x_delta, door_center_pos[1] + y_delta, -1.0]))

  @property
  def handle_contact_geoms(self):
      return[
        "handle_base",
        "handle",
      ]

  @property
  def door_contact_geoms(self):
      return[
        "door_box",
        "door_r_cyl",
        "door_l_cyl",
        "l_frame",
        "r_frame",
      ]

class DoorPullNoLatchRoomObject(MujocoXMLObject):
  """
  Door: pull with latch with walls
  """

  def __init__(self):
      super().__init__(xml_path_completion("objects/door_pull_no_latch_room.xml"))  

  def set_goal_xpos(self, x_delta, y_delta):
      """ Sets x,y position of goal site in door model with x and y offset from door center"""

      door_center_site = self.worldbody.find("./body/body/body/site[@name='door_center']")
      door_center_pos = string_to_array(door_center_site.get("pos"))
      goal_site = self.worldbody.find("./body/body/body/site[@name='goal']")
      goal_site.set("pos", array_to_string([door_center_pos[0] + x_delta, door_center_pos[1] + y_delta, -1.0]))

  @property
  def handle_contact_geoms(self):
      return[
        "handle_base",
        "handle",
      ]

  @property
  def door_contact_geoms(self):
      return[
        "door_box",
        "door_r_cyl",
        "door_l_cyl",
        "l_frame",
        "r_frame",
      ]

  @property
  def wall_contact_geoms(self):
      return[
        "wall_g0",
        "wall_g1",
        "wall_g2",
        "wall_g3",
      ]

class DoorPullNoLatchRoomWideObject(MujocoXMLObject):
  """
  Door: pull with no latch with walls
  """

  def __init__(self):
      super().__init__(xml_path_completion("objects/door_pull_no_latch_room_wide.xml"))  

  def set_goal_xpos(self, x_delta, y_delta):
      """ Sets x,y position of goal site in door model with x and y offset from door center"""

      door_center_site = self.worldbody.find("./body/body/body/site[@name='door_center']")
      door_center_pos = string_to_array(door_center_site.get("pos"))
      goal_site = self.worldbody.find("./body/body/body/site[@name='goal']")
      goal_site.set("pos", array_to_string([door_center_pos[0] + x_delta, door_center_pos[1] + y_delta, -1.0]))

  @property
  def handle_contact_geoms(self):
      return[
        "handle_base",
        "handle",
      ]

  @property
  def door_contact_geoms(self):
      return[
        "door_box",
        "door_r_cyl",
        "door_l_cyl",
        "l_frame",
        "r_frame",
      ]

  @property
  def wall_contact_geoms(self):
      return[
        "wall_g0",
        "wall_g1",
        "wall_g2",
        "wall_g3",
      ]

class EmptyWithGoalObject(MujocoXMLObject):
  """
  Empty arena with goal site
  """

  def __init__(self):
      super().__init__(xml_path_completion("objects/empty_with_goal.xml"))  

  def set_goal_xpos(self, x_delta, y_delta):
      """ Sets x,y position of goal site in door model with x and y offset from door center"""

      goal_site = self.worldbody.find("./body/body/site[@name='goal']")
      goal_site.set("pos", array_to_string([x_delta, y_delta, 0]))
