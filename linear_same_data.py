# coding=utf-8
import tensorflow as tf
import numpy as np

parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224"]

tf.app.flags.DEFINE_string("job_name", "", "輸入 'ps' 或是 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Job 的任務 index")
FLAGS = tf.app.flags.FLAGS


def main(_):

    cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        #train_X = np.linspace(-1.0, 1.0, 200)
        train_X = [ -1, -0.98994975, -0.9798995 , -0.96984925, -0.95979899,
                    -0.94974874, -0.93969849, -0.92964824, -0.91959799, -0.90954774,
                    -0.89949749, -0.88944724, -0.87939698, -0.86934673, -0.85929648,
                    -0.84924623, -0.83919598, -0.82914573, -0.81909548, -0.80904523,
                    -0.79899497, -0.78894472, -0.77889447, -0.76884422, -0.75879397,
                    -0.74874372, -0.73869347, -0.72864322, -0.71859296, -0.70854271,
                    -0.69849246, -0.68844221, -0.67839196, -0.66834171, -0.65829146,
                    -0.64824121, -0.63819095, -0.6281407 , -0.61809045, -0.6080402 ,
                    -0.59798995, -0.5879397 , -0.57788945, -0.5678392 , -0.55778894,
                    -0.54773869, -0.53768844, -0.52763819, -0.51758794, -0.50753769,
                    -0.49748744, -0.48743719, -0.47738693, -0.46733668, -0.45728643,
                    -0.44723618, -0.43718593, -0.42713568, -0.41708543, -0.40703518,
                    -0.39698492, -0.38693467, -0.37688442, -0.36683417, -0.35678392,
                    -0.34673367, -0.33668342, -0.32663317, -0.31658291, -0.30653266,
                    -0.29648241, -0.28643216, -0.27638191, -0.26633166, -0.25628141,
                    -0.24623116, -0.2361809 , -0.22613065, -0.2160804 , -0.20603015,
                    -0.1959799 , -0.18592965, -0.1758794 , -0.16582915, -0.15577889,
                    -0.14572864, -0.13567839, -0.12562814, -0.11557789, -0.10552764,
                    -0.09547739, -0.08542714, -0.07537688, -0.06532663, -0.05527638,
                    -0.04522613, -0.03517588, -0.02512563, -0.01507538, -0.00502513,
                     0.00502513,  0.01507538,  0.02512563,  0.03517588,  0.04522613,
                     0.05527638,  0.06532663,  0.07537688,  0.08542714,  0.09547739,
                     0.10552764,  0.11557789,  0.12562814,  0.13567839,  0.14572864,
                     0.15577889,  0.16582915,  0.1758794 ,  0.18592965,  0.1959799 ,
                     0.20603015,  0.2160804 ,  0.22613065,  0.2361809 ,  0.24623116,
                     0.25628141,  0.26633166,  0.27638191,  0.28643216,  0.29648241,
                     0.30653266,  0.31658291,  0.32663317,  0.33668342,  0.34673367,
                     0.35678392,  0.36683417,  0.37688442,  0.38693467,  0.39698492,
                     0.40703518,  0.41708543,  0.42713568,  0.43718593,  0.44723618,
                     0.45728643,  0.46733668,  0.47738693,  0.48743719,  0.49748744,
                     0.50753769,  0.51758794,  0.52763819,  0.53768844,  0.54773869,
                     0.55778894,  0.5678392 ,  0.57788945,  0.5879397 ,  0.59798995,
                     0.6080402 ,  0.61809045,  0.6281407 ,  0.63819095,  0.64824121,
                     0.65829146,  0.66834171,  0.67839196,  0.68844221,  0.69849246,
                     0.70854271,  0.71859296,  0.72864322,  0.73869347,  0.74874372,
                     0.75879397,  0.76884422,  0.77889447,  0.78894472,  0.79899497,
                     0.80904523,  0.81909548,  0.82914573,  0.83919598,  0.84924623,
                     0.85929648,  0.86934673,  0.87939698,  0.88944724,  0.89949749,
                     0.90954774,  0.91959799,  0.92964824,  0.93969849,  0.94974874,
                     0.95979899,  0.96984925,  0.9798995 ,  0.98994975,  1 ]        
        #train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0
        train_Y = [  8.3443785 ,  7.83161301,  7.98344043,  7.88636512,  8.30937155,
                     7.92718531,  7.5460088 ,  8.32882207,  8.29721388,  7.66769506,
                     8.28926573,  8.51019154,  8.2425172 ,  8.63217665,  8.39656248,
                     8.37932989,  8.3681118 ,  8.66231059,  8.74599727,  8.3086083 ,
                     8.22740911,  8.49360195,  7.80288305,  8.38396535,  8.44124186,
                     8.85993383,  8.24257343,  8.33919939,  8.05603968,  8.92746684,
                     8.04317407,  8.95270254,  8.50663858,  8.30855517,  8.54995921,
                     8.75084178,  8.76767397,  8.75640219,  8.75478782,  8.19470292,
                     8.67408394,  9.29895324,  8.62916222,  8.89501971,  8.57684251,
                     8.60657861,  8.380098  ,  8.7665455 ,  8.7970864 ,  8.05339427,
                     9.06871172,  9.66750183,  8.97354862,  8.85885375,  8.67858575,
                     9.09007117,  8.98755905,  8.84748811,  8.93063337,  9.78732998,
                     8.74850196,  9.07731033,  9.54484662,  9.19891283,  9.23784754,
                     9.83768732,  9.0765906 ,  9.67016712,  9.62440352,  9.1748678 ,
                     9.33321729,  9.33590937,  8.88322282,  9.57098016,  9.55207108,
                     9.53335571,  9.65295646,  9.86820264,  9.43049445,  9.63375118,
                    10.02316376,  9.28427452,  8.67907998,  9.63128977, 10.19055624,
                     9.8276527 ,  9.90665311, 10.08793422,  9.68173955,  9.80455805,
                     9.33688306,  9.06187324, 10.22522744,  9.83384692, 10.1945073 ,
                     9.58270806,  9.75751182, 10.11511491, 10.03200701,  9.54137282,
                    10.04768955,  9.67931075, 10.35233062, 10.25587241, 10.30106157,
                     9.94308508, 10.06193098, 10.28502168, 10.79548904, 10.53211298,
                    10.31796571, 10.15542595, 10.45403795,  9.97259096,  9.78854833,
                    10.54456292, 10.14682197, 10.37248411,  9.98801518, 10.49941381,
                    10.63239093, 10.03628831,  9.61677023, 10.61208486, 10.60015505,
                    10.61102889, 10.21480693, 10.95160192, 11.03146406, 10.6352423 ,
                    10.69868368, 10.8305434 , 10.85079345, 11.06108119, 10.57259787,
                    10.67038222, 10.74148979, 10.78872598, 10.40357608, 10.68747276,
                    10.92770974, 10.77462949, 10.92742624, 10.85820249, 10.75408114,
                    10.71826187, 11.03992898, 10.69787675, 10.79604096, 10.77665796,
                    11.47265982, 10.978255  , 10.67624327, 10.79456766, 11.27529177,
                    10.56216613, 11.39842318, 11.24799564, 11.48218086, 11.39288025,
                    10.37685339, 11.21776458, 11.4041129 , 10.98427994, 11.69002115,
                    11.57253837, 11.6084242 , 11.18402986, 11.02207051, 11.85652285,
                    11.09376878, 11.33124852, 11.40465921, 11.67844797, 11.50101417,
                    12.12781905, 11.85994554, 11.29616444, 11.95397829, 11.89468285,
                    11.48802303, 11.74499444, 11.36983767, 11.11365165, 11.43202979,
                    11.16640161, 11.47083374, 11.55555078, 11.59176805, 11.55293907,
                    11.79867157, 12.24373281, 12.10298262, 12.36499958, 11.7933149 ,
                    11.68314598, 12.44638094, 12.45268696, 11.96377907, 11.5824295 ]
        X = tf.placeholder("float")
        Y = tf.placeholder("float")

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            w = tf.Variable(0.0, name="weight")
            b = tf.Variable(0.0, name="bias")
            # 損失函式，用於描述模型預測值與真實值的差距大小，常見為`均方差(Mean Squared Error)`
            loss = tf.square(Y - tf.multiply(X, w) - b)

            global_step = tf.Variable(0)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # 建立 "Supervisor" 來負責監督訓練過程
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        with sv.managed_session(server.target) as sess:
            loss_value = 100
            turn = 1
            while not sv.should_stop() and loss_value > 70.0:
                # 執行一個非同步 training 步驟.
                # 若要執行同步可利用`tf.train.SyncReplicasOptimizer` 來進行
                for (x, y) in zip(train_X, train_Y):
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={X: x, Y: y})
                    print("步驟: {}".format(step))
                    print("Feed: ", x, y)
                loss_value = sess.run(loss, feed_dict={X: x, Y: y})
                print("------輪: {}, loss: {}------".format(turn, loss_value))
                turn += 1
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
