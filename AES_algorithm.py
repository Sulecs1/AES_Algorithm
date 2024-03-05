import base64
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


def pad(s):
    """
    Verilen diziyi AES blok boyutuna (16 byte) tamamlayacak şekilde doldurur.
    
    Args:
        s (bytes): Doldurulacak dizi
        
    Returns:
        bytes: Doldurulmuş dizi
    """
    return s + b"\0" * (AES.block_size - len(s) % AES.block_size)


def aes_encrypt(key, message):
    """
    Verilen anahtar ve mesajı AES algoritması kullanarak şifreler.

    Args:
        key (str): Şifreleme için kullanılacak anahtar
        message (str): Şifrelenecek mesaj

    Returns:
        str: Şifrelenmiş mesajın base64 kodlanmış hali
    """
    message = pad(bytes(message, 'utf-8'))
    #Başlatma vektörü her şifreleme için ayrı olmalı.Bundan dolayı random atanıyor.
    iv = Random.new().read(AES.block_size)
    #ES algoritmasının CBC (Cipher Block Chaining) modunu kullanacağını belirtir. CBC modunda, her bloğun şifrelenmesi sırasında önceki bloğun şifrelenmiş hali ile XOR işlemi yapılır.
    cipher = AES.new(bytes(key, 'utf-8'), AES.MODE_CBC, iv)
    cipher_data = base64.b64encode(iv + cipher.encrypt(message)).decode('utf-8')
    return cipher_data


def aes_decrypt(key, ciphertext):
    """
    Verilen anahtar ve şifreli metni AES algoritması kullanarak çözer.

    Args:
        key (str): Şifreleme için kullanılan anahtar
        ciphertext (str): Şifreli metin

    Returns:
        str: Çözülmüş metin
    """
    ciphertext = base64.b64decode(ciphertext)
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(bytes(key, 'utf-8'), AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext[AES.block_size:])
    #Çözülen metin, byte dizisinden rstrip(b"\0") yöntemi ile gereksiz boş karakterler (padding) kaldırılır.
    plaintext_data = plaintext.rstrip(b"\0").decode('utf-8')
    return plaintext_data

def udf_encrypt(key):
    """
    Verilen anahtar ile AES algoritmasını kullanarak sütunları şifrelemek için PySpark UDF oluşturur.

    Args:
        key (str): Şifreleme için kullanılan anahtar

    Returns:
        PySpark UDF: Şifreleme UDF'i
    """
    return F.udf(lambda text: aes_encrypt(key, text))


def udf_decrypt(key):
    """
    Verilen anahtar ile AES algoritmasını kullanarak şifreli sütunları çözmek için PySpark UDF oluşturur.

    Args:
        key (str): Şifreleme için kullanılan anahtar

    Returns:
        PySpark UDF: Şifre çözme UDF'i
    """
    return F.udf(lambda text: aes_decrypt(key, text))


# Spark Session oluşturma
spark = SparkSession.builder \
    .appName("AES Encryption Example") \
    .getOrCreate()

# Örnek veri oluşturma
data = [("1", "hello"), ("2", "world"), ("3", "spark")]

schema = StructType([
    StructField("id", StringType(), True),
    StructField("text", StringType(), True)
])

df = spark.createDataFrame(data, schema)

# Anahtar
key = '5ad8f52071d25165e7e68064ab56782'

# Şifreleme UDF'ini kullanarak 'text' sütununu şifreleme
df_encrypted = df.withColumn("encrypted_text", udf_encrypt(key)("text"))

# Şifreli veriyi gösterme
df_encrypted.show()

# Şifre çözme UDF'ini kullanarak şifreli metni çözme
df_decrypted = df_encrypted.withColumn("decrypted_text", udf_decrypt(key)("encrypted_text"))

# Çözülmüş veriyi gösterme
df_decrypted.show()

# Spark Session kapatma
spark.stop()
