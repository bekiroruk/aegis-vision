import unittest
from evaluate_segmentation import checked_rle, validate_export, NATIVE_CONFIDENCE

class Contracts(unittest.TestCase):
    def test_column_major_runs(self):
        value={"size":[3,3],"counts":[0,2,1,2,4]}
        self.assertEqual(checked_rle(value,3,3),value)
    def test_bad_runs(self):
        for counts in ([],[8],[-1,10],[True,8],[0.5,8.5],[10]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                checked_rle({"size":[3,3],"counts":counts},3,3)
    def test_wrong_shape(self):
        with self.assertRaises(ValueError): checked_rle({"size":[2,2],"counts":[4]},3,3)
    def fixture(self):
        source={"coco_image_id":7,"width":3,"height":3,"sha256":"a"}
        manifest={"images":[source],"classes":[{"label":"person","category_id":1}]}
        report={"version":1,"tracking":False,"confidence":.35,"mask_threshold":.5,"nms_iou":.45,"max_instances":100,
                "images":[{"id":7,"width":3,"height":3,"sha256":"a"}],
                "predictions":[{"image_id":7,"label":"person","score":.9,"segmentation":{"size":[3,3],"counts":[9]}}]}
        return manifest,report
    def test_empty_and_full_export(self):
        m,r=self.fixture();self.assertEqual(len(validate_export(m,r,{"person":1})[2]),1)
        r["predictions"]=[];self.assertEqual(validate_export(m,r,{"person":1})[2],[])
    def test_missing_image(self):
        m,r=self.fixture();r["images"]=[]
        with self.assertRaises(ValueError):validate_export(m,r,{"person":1})
    def test_protocol(self):
        for field,value in (("tracking",True),("confidence",.1),("max_instances",300)):
            m,r=self.fixture();r[field]=value
            with self.assertRaises(ValueError):validate_export(m,r,{"person":1})
    def test_score(self):
        m,r=self.fixture();r["predictions"][0]["score"]=NATIVE_CONFIDENCE
        self.assertEqual(len(validate_export(m,r,{"person":1})[2]),1)
        for score in (float("nan"),1.1,-1,True,NATIVE_CONFIDENCE-1e-8):
            m,r=self.fixture();r["predictions"][0]["score"]=score
            with self.assertRaises(ValueError):validate_export(m,r,{"person":1})
    def test_hash_and_mapping(self):
        m,r=self.fixture();r["images"][0]["sha256"]="b"
        with self.assertRaises(ValueError):validate_export(m,r,{"person":1})
        m,r=self.fixture()
        with self.assertRaises(ValueError):validate_export(m,r,{"person":2})
    def test_instance_bound(self):
        m,r=self.fixture();r["predictions"]*=101
        with self.assertRaises(ValueError):validate_export(m,r,{"person":1})

if __name__ == "__main__":unittest.main()
