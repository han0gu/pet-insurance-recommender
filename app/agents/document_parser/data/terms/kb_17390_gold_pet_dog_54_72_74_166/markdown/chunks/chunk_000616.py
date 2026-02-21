from langchain_core.documents import Document

chunk = Document(
    page_content=('| 관 련 법 규 수의사법 제12조(진단서 등) ① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증명서 또는 '
 '처방전을 발급하지 못하며, 「약사법」 제85조제6항에 따른 동물용 의약품(이하 "동물용 의약품"이라 한다)을 처방·투약하지 못한다. '
 '다만, 직접 진료하거나 검안한 수의사가 부득이한 사유로 진단서, 검안서 또는 증명서를 발급할 수 없을 때에는 같은 동물병원에 종사하는 '
 '다른 수의사가 진료부 등에 의하여 발급할 수 있다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
