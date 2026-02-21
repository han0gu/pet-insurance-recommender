from langchain_core.documents import Document

chunk = Document(
    page_content=('사고가 그 증상을 악화시킨<br>통<br>부분만큼, 즉 이 사고와의 관여도를 산정하여 평가한다.<br>4) 추간판탈출증으로 인한 신경 '
 '장해는 수술 또는 시술(비수술적 치료) 후 사항<br>6개월 이상 지난 후에 평가한다.<br>5) 신경학적 검사상 나타난 저린감이나 '
 '방사통 등 신경자극증상의 원인으로<br>CT, MRI 등 영상검사에서 추간판탈출증이 확인된 경우를 추간판탈출증으<br>로 진단하며, 수술 '
 '여부에 관계없이 운동장해 및 기형장해로 평가하지 보<br>않는다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001552',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
