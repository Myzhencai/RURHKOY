import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

class freezeGraph():
    def __init__(self, model_dir, output_node_names):
        self.model_dir = model_dir
        self.output_node_names = output_node_names

    def freeze_graph(self):
        """Extract the sub graph defined by the output nodes and convert 
        all its variables into constant 

        Args:
            model_dir: the root folder containing the checkpoint state file
            output_node_names: a string, containing all the output node's names, 
                                comma separated
        """
        if not tf.io.gfile.exists(self.model_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % self.model_dir)

        if not self.output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1

        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        input_checkpoint = self.model_dir + "model.ckpt"
        #print (input_checkpoint)
        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = self.model_dir + "/frozen_model.pb"

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We start a session using a temporary fresh Graph
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We restore the weights
            saver.restore(sess, input_checkpoint)
            #print ([n.name for n in tf.get_default_graph().as_graph_def().node])
            #for op in tf.get_default_graph().get_operations():
              #print(str(op.name))
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.compat.v1.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                self.output_node_names.split(",") # The output node names are used to select the usefull nodes
            ) 

            # Finally we serialize and dump the output graph to the filesystem
            with tf.io.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            print ([n.name for n in output_graph_def.node])
        return output_graph_def