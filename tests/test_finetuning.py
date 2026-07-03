import unittest
import tensorflow as tf
from papa_disease.config import CLASS_NAMES, IMAGE_SIZE
from papa_disease.training import build_model, build_finetune_model

class TestFinetuning(unittest.TestCase):
    def test_build_finetune_model_freezes_and_unfreezes_correctly(self):
        # 1. Crear el modelo base original
        model = build_model()
        
        # Al inicio, el base_model (resnet50) no es entrenable
        resnet_layer = next((l for l in model.layers if l.name == "resnet50"), None)
        self.assertIsNotNone(resnet_layer)
        self.assertFalse(resnet_layer.trainable)
        
        # 2. Aplicar build_finetune_model
        finetuned_model = build_finetune_model(model)
        
        # Despues del finetuning, el base_model es entrenable
        resnet_layer = next((l for l in finetuned_model.layers if l.name == "resnet50"), None)
        self.assertTrue(resnet_layer.trainable)
        
        # Pero no todas las subcapas son entrenables. Comprobamos las ultimas 30.
        trainable_count = sum(1 for layer in resnet_layer.layers if layer.trainable)
        self.assertGreater(trainable_count, 0, "Deberia haber capas entrenables en resnet50")
        self.assertLess(trainable_count, len(resnet_layer.layers), "No todas las capas deberian ser entrenables")
        
        # 3. Comprobar compilacion
        self.assertEqual(finetuned_model.loss, "sparse_categorical_crossentropy")
        self.assertTrue(isinstance(finetuned_model.optimizer, tf.keras.optimizers.Adam))
        
        # La tasa de aprendizaje deberia ser baja (1e-5)
        # Note: In tf/keras, learning_rate can be a tensor or scalar. We check its approximate value.
        lr = finetuned_model.optimizer.learning_rate.numpy()
        self.assertAlmostEqual(lr, 1e-5, places=6)

if __name__ == "__main__":
    unittest.main()
