#!/usr/bin/env bash
# Cài và chạy HDFS một máy (pseudo-distributed) trên macOS, rồi đưa 7 file CSV lên HDFS.
# Chạy trong thư mục credit-risk-spark:   bash scripts/hdfs_setup.sh
set -e

export JAVA_HOME="${JAVA_HOME:-$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home}"
command -v hdfs >/dev/null || brew install hadoop

HADOOP_HOME="$(brew --prefix hadoop)/libexec"
CONF="$HADOOP_HOME/etc/hadoop"
DATA="$HOME/hadoop-data"
mkdir -p "$DATA/namenode" "$DATA/datanode"

cat > "$CONF/core-site.xml" <<XML
<?xml version="1.0"?>
<configuration>
  <property><name>fs.defaultFS</name><value>hdfs://localhost:9000</value></property>
</configuration>
XML

cat > "$CONF/hdfs-site.xml" <<XML
<?xml version="1.0"?>
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://$DATA/namenode</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://$DATA/datanode</value></property>
</configuration>
XML

grep -q "^export JAVA_HOME=" "$CONF/hadoop-env.sh" || echo "export JAVA_HOME=\"$JAVA_HOME\"" >> "$CONF/hadoop-env.sh"

# Định dạng NameNode lần đầu
[ -d "$DATA/namenode/current" ] || hdfs namenode -format -nonInteractive

# Khởi động NameNode và DataNode (không cần SSH)
hdfs --daemon start namenode || true
hdfs --daemon start datanode || true
sleep 5
jps | grep -E "NameNode|DataNode" || { echo "HDFS chưa chạy, xem log trong $HADOOP_HOME/logs"; exit 1; }

# Đưa dữ liệu lên HDFS
hdfs dfs -mkdir -p /credit-risk/raw
for f in application_train bureau bureau_balance previous_application POS_CASH_balance installments_payments credit_card_balance; do
  echo "Đưa $f.csv lên HDFS..."
  hdfs dfs -put -f "data/$f.csv" /credit-risk/raw/
done
hdfs dfs -ls -h /credit-risk/raw
echo
echo "Xong. Giao diện HDFS: http://localhost:9870  (Utilities → Browse the file system)"
echo "Tắt HDFS khi không dùng:  hdfs --daemon stop datanode && hdfs --daemon stop namenode"
