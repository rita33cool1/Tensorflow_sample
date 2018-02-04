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

        train_X = np.linspace(-1.0, 1.0, 100)
        #train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0
        train_Y = [ 7.54979061, 8.23769983, 7.69990378, 7.70666437, 8.23783833, 8.46157815,
                    8.34597745, 8.56917846, 9.16073336, 8.98375415, 8.52265534, 8.28596715,
                    8.68673277, 8.24133037, 8.50903657, 8.78558513, 8.98083051, 7.99413078,
                    8.50809117, 8.5247203, 9.3456627, 8.73630572, 8.86427564, 9.24927778,
                    8.87832333, 9.17925754, 8.755319, 8.6201733, 8.85415126, 9.27769285,
                    8.85226867, 9.35896285, 9.31925512, 9.49557716, 9.56481763, 9.36132552,
                    9.44727506, 9.29612536, 9.18367499, 9.62836374, 9.13952843, 9.74083392,
                    9.86173934, 9.47037291, 9.57127795, 9.71458821, 10.03056769, 10.25280349, 
                    9.45081567, 9.7782832, 9.77692766, 10.33587958, 10.64136778, 9.92788176,
                    9.84699949, 10.52375653, 9.98935615, 9.8772447, 11.1737163, 10.29267264,
                    10.25476844, 10.74833561, 10.77331738, 10.15864171, 10.10233281, 10.84521747,
                    11.21850992, 10.05172119, 10.42862251, 10.08065297, 10.843441, 10.87607787,
                    10.71915812, 11.14718661, 10.81289518, 10.75835296, 10.971318, 11.71847065,
                    11.37763486, 10.98270058, 11.59164822, 11.01465886, 10.82485681, 11.69351235,
                    11.30956068, 11.84618687, 11.62343215, 11.0797436, 11.27897043, 11.98335276,
                    12.13022781, 11.49475733, 11.99216458, 11.51510155, 11.74086278, 11.96880742,
                    12.06748988, 12.16443447, 12.0903128, 11.94404893]
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

            global_step = tf.Variable(0).minimize(loss, global_step=global_step)
            train_op = tf.train.AdagradOptimizer(0.1).minimize(loss, global_step=global_step)
            #train_op = tf.train.AdagradOptimizer(0.1)
            #train_op = tf.train.SyncReplicasOptimizer(train_op, 
            #                                          replicas_to_aggregate=len(workers), 
            #                                          total_num_replicas=len(workers),
            #                                          use_locking=True
            #                                         ).minimize(loss, global_step=global_step)
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
            while not sv.should_stop() and loss_value > 70.0:
                # 執行一個非同步 training 步驟.
                # 若要執行同步可利用`tf.train.SyncReplicasOptimizer` 來進行
                for (x, y) in zip(train_X, train_Y):
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={X: x, Y: y})

                loss_value = sess.run(loss, feed_dict={X: x, Y: y})
                print("步驟: {}, loss: {}".format(step, loss_value))

        sv.stop()


if __name__ == "__main__":
    tf.app.run()
