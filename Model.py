class ImageModel():
    def __init__(self):
        self.input_words = np.load('input_words.npy')
        self.train_label = np.load('train_label.npy')
        self.vertor = np.load('labels_train.npy')
        self.test = np.load('labels_test.npy')

        with open('rev_dictionary_.txt', 'r') as f:
            self.rev_dictionary_ = eval(f.read())
        with open('dictionary_.txt', 'r') as g:
            self.dictionary_ = eval(g.read())

        self.weights = {
            'wc1': tf.get_variable('weight1', shape=[3, 3, 3, 32], initializer=tf.glorot_normal_initializer()),
            'wc2': tf.get_variable('weight2', shape=[3, 3, 32, 64], initializer=tf.glorot_normal_initializer()),
            'wc3': tf.get_variable('weight3', shape=[3, 3, 64, 128], initializer=tf.glorot_normal_initializer()),
            'wc4': tf.get_variable('weight4', shape=[3, 3, 128, 256], initializer=tf.glorot_normal_initializer()),
            'wd1': tf.get_variable('weightd1', shape=[9216, 512], initializer=tf.glorot_normal_initializer()),
            'wd2': tf.get_variable('weightd2', shape=[512, 261], initializer=tf.glorot_normal_initializer())}
        self.bias = {
            'bc1': tf.get_variable('bias1', initializer=tf.zeros([32], dtype=tf.float32)),
            'bc2': tf.get_variable('bias2', initializer=tf.zeros([64], dtype=tf.float32)),
            'bc3': tf.get_variable('bias3', initializer=tf.zeros([128], dtype=tf.float32)),
            'bc4': tf.get_variable('bias4', initializer=tf.zeros([256],  dtype=tf.float32)),
            'bd1': tf.get_variable('biasd1', initializer=tf.zeros([512], dtype=tf.float32)),
            'bd2': tf.get_variable('biasd2', initializer=tf.zeros([261], dtype=tf.float32))}
        self.strides = {
            'sc1': [1, 1, 1, 1],
            'sc2': [1, 1, 1, 1],
            'sc3': [1, 1, 1, 1],
            'sc4': [1, 1, 1, 1],
            'sp1': [1, 2, 2, 1],
            'sp2': [1, 2, 2, 1],
            'sp3': [1, 2, 2, 1],
            'sp4': [1, 2, 2, 1]}
        self.pooling_size = {
            'kp1': [1, 2, 2, 1],
            'kp2': [1, 2, 2, 1],
            'kp3': [1, 2, 2, 1],
            'kp4': [1, 2, 2, 1]}

    def conv(self, x, kernel, strides, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, kernel, strides, padding='VALID'), b))

    def max_pooling(self, x, kernel, strides):
        return tf.nn.max_pool(x, kernel, strides, padding='VALID')

    def fc(self, x, w, b):
        return tf.nn.relu(tf.add(tf.matmul(x, w), b))

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        return score

    def getBatch(self, batch_size):
        x = imagetrain_data
        y = self.vertor
        index = [i for i in range(len(x))]
        random.shuffle(index)
        x1 = x[index]
        y1 = y[index]
        for i in range(0, len(x1), batch_size):
            x2 = x1[i:i+batch_size]
            y2 = y1[i:i+batch_size]
            print("Batchs:{0}/{1}".format(i, len(x1)))
            yield x2, y2

    def getTestBatch(self, batch_size):
        x = imagetest_data
        y = self.test
        index = [i for i in range(len(x))]
        random.shuffle(index)
        x1 = x[index]
        y1 = y[index]
        for i in range(0, len(x1), batch_size):
            x3 = x1[i:i+batch_size]
            y3 = y1[i:i+batch_size]
            yield x3, y3

    def buildModel(self, inputs, weights, bias, strides, pooling_size):
        with tf.name_scope('conv1'):
            conv1 = self.conv(inputs, weights['wc1'], strides['sc1'], bias['bc1'])
            tf.summary.histogram('conv1' + '/weights', weights['wc1'])
            tf.summary.histogram('conv1' + '/bias', bias['bc1'])
        with tf.name_scope('pool1'):
            pool1 = self.max_pooling(conv1, pooling_size['kp1'], strides['sp1'])

        with tf.name_scope('conv2'):
            conv2 = self.conv(pool1, weights['wc2'], strides['sc2'], bias['bc2'])
            tf.summary.histogram('conv2' + '/weights', weights['wc2'])
            tf.summary.histogram('conv2' + '/bias', bias['bc2'])
        with tf.name_scope('pool2'):
            pool2 = self.max_pooling(conv2, pooling_size['kp2'], strides['sp2'])

        with tf.name_scope('conv3'):
            conv3 = self.conv(pool2, weights['wc3'], strides['sc3'], bias['bc3'])
            tf.summary.histogram('conv3' + '/weights', weights['wc3'])
            tf.summary.histogram('conv3' + '/bias', bias['bc3'])
        with tf.name_scope('pool3'):
            pool3 = self.max_pooling(conv3, pooling_size['kp3'], strides['sp3'])

        with tf.name_scope('conv4'):
            conv4 = self.conv(pool3, weights['wc4'], strides['sc4'], bias['bc4'])
            tf.summary.histogram('conv4' + '/weights', weights['wc4'])
            tf.summary.histogram('conv4' + '/bias', bias['bc4'])
        with tf.name_scope('pool4'):
            pool4 = self.max_pooling(conv4, pooling_size['kp4'], strides['sp4'])

        flatten = tf.reshape(pool4, [-1, 9216])

        with tf.name_scope('fc1'):
            fc1 = self.fc(flatten, weights['wd1'], bias['bd1'])
            tf.summary.histogram('fc1' + '/weights', weights['wd1'])
            tf.summary.histogram('fc1' + '/bias', bias['bd1'])

        fc1_dropout = tf.nn.dropout(fc1, 1)
        with tf.name_scope('fc2'):
            outputs = tf.add(tf.matmul(fc1_dropout, weights['wd2']), bias['bd2'])
            tf.summary.histogram('fc2' + '/weights', weights['wd2'])
            tf.summary.histogram('fc2' + '/bias', bias['bd2'])
            tf.summary.histogram('fc2' + '/output', outputs)
        return outputs

    def trainModel(self, output, lr, image_size=128, batct_size=1, epochs=10):
        x = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='inputX')
        y = tf.placeholder(tf.float32, [None, output], name='outputY')
        tf.summary.image('input', x)

        pred = self.buildModel(x, self.weights, self.bias, self.strides, self.pooling_size)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)

        tf_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('train_loss', tf_loss)
        cos = tf.reduce_mean(self.cosine(y, tf.sigmoid(pred)))
        tf.summary.scalar('train_accuracy', cos)

        test_loss_ = tf.reduce_mean(cross_entropy)
        test_cos_ = tf.reduce_mean(self.cosine(y, tf.sigmoid(pred)))

        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=100)
        posi = 'CNN/'
        with tf.Session() as sess:
            test_loss1 = []
            test_acc1 = []
            sess.run(init)
            merge = tf.summary.merge_all()
            print('---------------Start Train------------------')
            writer = tf.summary.FileWriter(posi+"logs_train/", sess.graph)
            for epoch in range(1, epochs+1):
                batchs = self.getBatch(batct_size)
                for x1, y1 in batchs:
                    loss_, _, rs, acc_ = sess.run([tf_loss, train_step, merge, cos], feed_dict={x: x1, y: y1})
                    print('Epoch: {0} -----~~~~~~----> Accuracy==>{1}, Loss==>{2}'.format(epoch, acc_, loss_))

                batch_test = self.getTestBatch(50)
                for x2, y2 in batch_test:
                    test_loss, test_acc = sess.run([test_loss_, test_cos_], feed_dict={x: x2, y: y2})
                    test_loss1.append(test_loss)
                    test_acc1.append(test_acc)
                    print('Epoch: {0} --------  ----> Test_Accuracy==>{1}, Test_Loss==>{2}'.format(epoch, test_acc,
                                                                                                   test_loss))
                if epoch % 1 == 0:
                    saver.save(sess, posi+"model_train/image_loss{0}".format(epoch))
                writer.add_summary(rs, epoch)
                np.save(posi + 'test_loss.npy', test_loss1)
                np.save(posi + 'test_acc.npy', test_acc1)
